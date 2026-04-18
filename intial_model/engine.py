from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .dataset import SequenceRecord
from .detector import YoloFallbackDetector
from .metrics import IQAScorer, SequenceMetrics, aggregate_metrics, evaluate_sequence
from .restoration import restore_frame
from .utils import (
    clip_box_xywh,
    compute_square_crop,
    crop_normalized_to_box,
    crop_square,
    ensure_dir,
    save_json,
)


@dataclass
class TrainEpochResult:
    loss: float
    box_loss: float
    score_loss: float


def _box_iou_tensor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x1 = pred[:, 0] - (pred[:, 2] / 2.0)
    pred_y1 = pred[:, 1] - (pred[:, 3] / 2.0)
    pred_x2 = pred[:, 0] + (pred[:, 2] / 2.0)
    pred_y2 = pred[:, 1] + (pred[:, 3] / 2.0)

    target_x1 = target[:, 0] - (target[:, 2] / 2.0)
    target_y1 = target[:, 1] - (target[:, 3] / 2.0)
    target_x2 = target[:, 0] + (target[:, 2] / 2.0)
    target_y2 = target[:, 1] + (target[:, 3] / 2.0)

    inter_x1 = torch.maximum(pred_x1, target_x1)
    inter_y1 = torch.maximum(pred_y1, target_y1)
    inter_x2 = torch.minimum(pred_x2, target_x2)
    inter_y2 = torch.minimum(pred_y2, target_y2)

    inter = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
    pred_area = (pred_x2 - pred_x1).clamp(min=1e-6) * (pred_y2 - pred_y1).clamp(min=1e-6)
    target_area = (target_x2 - target_x1).clamp(min=1e-6) * (target_y2 - target_y1).clamp(min=1e-6)
    union = pred_area + target_area - inter + 1e-6
    return inter / union


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    channels_last: bool = True,
    scaler: GradScaler | None = None,
) -> TrainEpochResult:
    model.train()
    total_loss = 0.0
    total_box_loss = 0.0
    total_score_loss = 0.0
    criterion = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss()

    if scaler is None:
        scaler = GradScaler(device=device.type, enabled=use_amp and device.type == "cuda")

    for batch in tqdm(loader, desc="train", leave=False):
        template = batch["template"].to(device, non_blocking=True)
        search = batch["search"].to(device, non_blocking=True)
        template_restored = batch["template_restored"].to(device, non_blocking=True)
        search_restored = batch["search_restored"].to(device, non_blocking=True)
        target_box = batch["target_box"].to(device)
        condition = batch["condition"].to(device)
        if channels_last:
            template = template.contiguous(memory_format=torch.channels_last)
            search = search.contiguous(memory_format=torch.channels_last)
            template_restored = template_restored.contiguous(memory_format=torch.channels_last)
            search_restored = search_restored.contiguous(memory_format=torch.channels_last)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred_box, pred_score_logits = model(template, template_restored, search, search_restored, condition)
            iou = _box_iou_tensor(pred_box, target_box).detach()
            target_score = (iou > 0.3).float()
            box_loss = criterion(pred_box, target_box) + (1.0 - _box_iou_tensor(pred_box, target_box).mean())
            score_loss = bce(pred_score_logits, target_score)
            loss = box_loss + (0.25 * score_loss)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_box_loss += box_loss.item()
        total_score_loss += score_loss.item()

    count = max(1, len(loader))
    return TrainEpochResult(
        loss=total_loss / count,
        box_loss=total_box_loss / count,
        score_loss=total_score_loss / count,
    )


@torch.no_grad()
def run_tracker_on_sequence(
    model: nn.Module,
    dataset_root: Path,
    sequence: SequenceRecord,
    device: torch.device,
    template_size: int = 128,
    search_size: int = 256,
    template_scale: float = 2.0,
    search_scale: float = 4.5,
    update_template_every: int = 15,
    confidence_blend: float = 0.65,
    use_restoration: bool = True,
    detector: YoloFallbackDetector | None = None,
    yolo_trigger_threshold: float = 0.25,
    yolo_cooldown: int = 12,
) -> tuple[list[list[float]], list[float]]:
    model.eval()
    image0 = Image.open(dataset_root / sequence.image_paths[0]).convert("RGB")
    width, height = image0.size
    current_box = [float(v) for v in sequence.gt_rect[0]]
    predictions = [current_box.copy()]
    scores = [1.0]
    condition = torch.tensor([0 if sequence.condition == "haze" else 1], dtype=torch.long, device=device)

    t_cx, t_cy, t_side = compute_square_crop(current_box, template_scale)
    template_crop, _ = crop_square(image0, t_cx, t_cy, t_side, template_size)
    template_restored_crop = restore_frame(template_crop, sequence.condition) if use_restoration else template_crop
    template_tensor = TF.to_tensor(template_crop).unsqueeze(0).to(device)
    template_restored_tensor = TF.to_tensor(template_restored_crop).unsqueeze(0).to(device)
    last_detector_frame = -yolo_cooldown

    for frame_idx in range(1, len(sequence.image_paths)):
        image = Image.open(dataset_root / sequence.image_paths[frame_idx]).convert("RGB")
        s_cx, s_cy, s_side = compute_square_crop(current_box, search_scale)
        search_crop, crop_info = crop_square(image, s_cx, s_cy, s_side, search_size)
        search_restored_crop = restore_frame(search_crop, sequence.condition) if use_restoration else search_crop
        search_tensor = TF.to_tensor(search_crop).unsqueeze(0).to(device)
        search_restored_tensor = TF.to_tensor(search_restored_crop).unsqueeze(0).to(device)

        pred_box_norm, pred_score_logits = model(
            template_tensor,
            template_restored_tensor,
            search_tensor,
            search_restored_tensor,
            condition,
        )
        pred_box = crop_normalized_to_box(pred_box_norm[0].detach().cpu().tolist(), crop_info, image.size)
        score = float(torch.sigmoid(pred_score_logits)[0].item())

        if score < 0.35:
            blend = confidence_blend * score
            pred_box = clip_box_xywh(
                [
                    ((1.0 - blend) * current_box[0]) + (blend * pred_box[0]),
                    ((1.0 - blend) * current_box[1]) + (blend * pred_box[1]),
                    ((1.0 - blend) * current_box[2]) + (blend * pred_box[2]),
                    ((1.0 - blend) * current_box[3]) + (blend * pred_box[3]),
                ],
                width=image.size[0],
                height=image.size[1],
            )

        if (
            detector is not None
            and score < yolo_trigger_threshold
            and (frame_idx - last_detector_frame) >= yolo_cooldown
        ):
            det = detector.detect_best(image, current_box)
            if det is not None:
                pred_box = clip_box_xywh(det.xywh, width=image.size[0], height=image.size[1])
                score = max(score, 0.55 * det.confidence)
                last_detector_frame = frame_idx

        current_box = pred_box
        predictions.append(current_box.copy())
        scores.append(score)

        if score > 0.6 and (frame_idx % update_template_every == 0):
            t_cx, t_cy, t_side = compute_square_crop(current_box, template_scale)
            template_crop, _ = crop_square(image, t_cx, t_cy, t_side, template_size)
            template_restored_crop = restore_frame(template_crop, sequence.condition) if use_restoration else template_crop
            template_tensor = TF.to_tensor(template_crop).unsqueeze(0).to(device)
            template_restored_tensor = TF.to_tensor(template_restored_crop).unsqueeze(0).to(device)

    return predictions, scores


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset_root: Path,
    sequences: list[SequenceRecord],
    device: torch.device,
    output_predictions_path: Path | None = None,
    use_restoration: bool = True,
    yolo_fallback: bool = False,
    yolo_model_name: str = "yolov8n.pt",
    iqa_cache_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, list[list[float]] | list[float]]]]:
    per_sequence: list[SequenceMetrics] = []
    prediction_payload: dict[str, dict[str, list[list[float]] | list[float]]] = {}
    iqa_scorer = IQAScorer(cache_path=iqa_cache_path)
    all_image_paths: list[str] = []
    for sequence in sequences:
        all_image_paths.extend([str(dataset_root / image_path) for image_path in sequence.image_paths])
    iqa_scorer.warmup(all_image_paths)
    detector = YoloFallbackDetector(model_name=yolo_model_name, device=device.type) if yolo_fallback else None

    for sequence in tqdm(sequences, desc="eval", leave=False):
        pred_boxes, scores = run_tracker_on_sequence(
            model,
            dataset_root,
            sequence,
            device,
            use_restoration=use_restoration,
            detector=detector,
        )
        frame_iqas = [iqa_scorer.iqa(str(dataset_root / image_path)) for image_path in sequence.image_paths]
        metrics = evaluate_sequence(pred_boxes, [[float(v) for v in box] for box in sequence.gt_rect], frame_iqas)
        per_sequence.append(metrics)
        prediction_payload[sequence.video_dir] = {
            "pred_rect": [[round(v, 3) for v in box] for box in pred_boxes],
            "confidence": [round(v, 5) for v in scores],
        }

    aggregate = aggregate_metrics(per_sequence)
    iqa_scorer.flush()
    if output_predictions_path is not None:
        save_json(output_predictions_path, prediction_payload)
    return aggregate, prediction_payload


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    output_dir: Path,
    model_name: str,
) -> Path:
    ensure_dir(output_dir)
    filename = (
        f"{model_name}_epoch{epoch:02d}_"
        f"iou{metrics['mean_iou']:.4f}_"
        f"qp{metrics['qp']:.4f}.pt"
    )
    checkpoint_path = output_dir / filename
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_name": model_name,
        },
        checkpoint_path,
    )
    return checkpoint_path
