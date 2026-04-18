from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from vistac_tracker.dataset import TrackingPairDataset, load_sequences
from vistac_tracker.engine import evaluate_model, save_checkpoint, train_one_epoch
from vistac_tracker.models import build_model
from vistac_tracker.utils import ensure_dir, load_json, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact trackers for the ExtremeTrack dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/vistac/ExtremeTrack"))
    parser.add_argument("--output-root", type=Path, default=Path("D:/vistac/outputs"))
    parser.add_argument("--models", nargs="+", default=["mixformer_lite", "mixformer_lite_large"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-epoch", type=int, default=9000)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=0)
    parser.add_argument("--condition", choices=["all", "haze", "rain"], default="all")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--use-restoration", action="store_true", default=True)
    parser.add_argument("--yolo-fallback", action="store_true", default=False)
    parser.add_argument("--yolo-model-name", type=str, default="yolov8n.pt")
    parser.add_argument("--iqa-cache-path", type=Path, default=Path("D:/vistac/outputs/cache/val_iqa_cache.json"))
    parser.add_argument("--compile-model", action="store_true", default=False)
    parser.add_argument("--train-bench", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available. The training script is configured for GPU use.")
    if args.train_bench:
        cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    output_root = ensure_dir(args.output_root)
    experiments_root = ensure_dir(output_root / "experiments")
    checkpoints_root = ensure_dir(output_root / "checkpoints")
    predictions_root = ensure_dir(output_root / "predictions")

    condition_filter = None if args.condition == "all" else args.condition
    train_dataset = TrackingPairDataset(
        dataset_root=args.dataset_root,
        annotation_name="ExtremeTrack_train.json",
        condition_filter=condition_filter,
        samples_per_epoch=args.samples_per_epoch,
    )
    val_sequences = load_sequences(args.dataset_root, "ExtremeTrack_val.json")
    if condition_filter:
        val_sequences = [s for s in val_sequences if s.condition == condition_filter]
    if args.max_val_sequences > 0:
        val_sequences = val_sequences[: args.max_val_sequences]

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    leaderboard_path = output_root / "leaderboard.json"
    all_results: dict[str, dict[str, float | str | int]] = load_json(leaderboard_path) if leaderboard_path.exists() else {}

    for model_name in args.models:
        model = build_model(model_name).to(device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if args.compile_model:
            model = torch.compile(model, mode="max-autotune")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(device=device.type, enabled=args.use_amp and device.type == "cuda")
        best_qp = -1.0
        best_checkpoint = None
        history: list[dict[str, float | int | str]] = []

        for epoch in range(1, args.epochs + 1):
            if args.max_train_batches > 0:
                truncated_loader = []
                for batch_idx, batch in enumerate(loader):
                    truncated_loader.append(batch)
                    if batch_idx + 1 >= args.max_train_batches:
                        break
                train_result = train_one_epoch(
                    model,
                    truncated_loader,
                    optimizer,
                    device,
                    use_amp=args.use_amp,
                    channels_last=args.channels_last,
                    scaler=scaler,
                )
            else:
                train_result = train_one_epoch(
                    model,
                    loader,
                    optimizer,
                    device,
                    use_amp=args.use_amp,
                    channels_last=args.channels_last,
                    scaler=scaler,
                )

            prediction_path = predictions_root / f"{model_name}_epoch{epoch:02d}_val_predictions.json"
            metrics, _ = evaluate_model(
                model,
                args.dataset_root,
                val_sequences,
                device,
                prediction_path,
                use_restoration=args.use_restoration,
                yolo_fallback=args.yolo_fallback,
                yolo_model_name=args.yolo_model_name,
                iqa_cache_path=args.iqa_cache_path,
            )
            checkpoint_path = save_checkpoint(model, optimizer, epoch, metrics, checkpoints_root, model_name)

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_box_loss": train_result.box_loss,
                "train_score_loss": train_result.score_loss,
                **metrics,
                "checkpoint": str(checkpoint_path),
                "prediction_file": str(prediction_path),
            }
            history.append(epoch_result)

            if metrics["qp"] > best_qp:
                best_qp = metrics["qp"]
                best_checkpoint = checkpoint_path

        all_results[model_name] = {
            "best_qp": best_qp,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
            "epochs": args.epochs,
            "condition": args.condition,
        }
        save_json(experiments_root / f"{model_name}_history.json", {"history": history, "summary": all_results[model_name]})

    save_json(leaderboard_path, all_results)


if __name__ == "__main__":
    main()
