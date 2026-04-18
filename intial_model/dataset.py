from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .restoration import restore_frame
from .utils import box_to_crop_normalized, compute_square_crop, crop_square, load_json


@dataclass
class SequenceRecord:
    name: str
    video_dir: str
    image_paths: list[str]
    gt_rect: list[list[float]]
    condition: str


def load_sequences(dataset_root: Path, annotation_name: str) -> list[SequenceRecord]:
    annotations = load_json(dataset_root / annotation_name)
    records: list[SequenceRecord] = []
    for name, seq in annotations.items():
        condition = "haze" if name.lower().endswith("haze") else "rain"
        records.append(
            SequenceRecord(
                name=name,
                video_dir=seq["video_dir"],
                image_paths=seq["img_names"],
                gt_rect=seq["gt_rect"],
                condition=condition,
            )
        )
    return records


class TrackingPairDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        annotation_name: str,
        template_size: int = 128,
        search_size: int = 256,
        template_scale: float = 2.0,
        search_scale: float = 4.5,
        samples_per_epoch: int = 8000,
        max_frame_gap: int = 12,
        condition_filter: str | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.records = load_sequences(dataset_root, annotation_name)
        if condition_filter:
            self.records = [r for r in self.records if r.condition == condition_filter]
        self.template_size = template_size
        self.search_size = search_size
        self.template_scale = template_scale
        self.search_scale = search_scale
        self.samples_per_epoch = samples_per_epoch
        self.max_frame_gap = max_frame_gap

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _open_image(self, rel_path: str) -> Image.Image:
        return Image.open(self.dataset_root / rel_path).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = random.choice(self.records)
        upper = len(record.image_paths) - 1
        search_idx = random.randint(1, upper)
        template_idx = random.randint(max(0, search_idx - self.max_frame_gap), search_idx - 1)

        template_image = self._open_image(record.image_paths[template_idx])
        search_image = self._open_image(record.image_paths[search_idx])
        template_box = [float(v) for v in record.gt_rect[template_idx]]
        search_box = [float(v) for v in record.gt_rect[search_idx]]

        t_cx, t_cy, t_side = compute_square_crop(template_box, self.template_scale)
        s_cx, s_cy, s_side = compute_square_crop(template_box, self.search_scale)

        template_crop, _ = crop_square(template_image, t_cx, t_cy, t_side, self.template_size)
        search_crop, search_crop_info = crop_square(search_image, s_cx, s_cy, s_side, self.search_size)

        target = box_to_crop_normalized(search_box, search_crop_info)

        condition_name = "haze" if record.condition == "haze" else "rain"
        template_restored = restore_frame(template_crop, condition_name)
        search_restored = restore_frame(search_crop, condition_name)

        template_tensor = TF.to_tensor(template_crop)
        search_tensor = TF.to_tensor(search_crop)
        template_restored_tensor = TF.to_tensor(template_restored)
        search_restored_tensor = TF.to_tensor(search_restored)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        condition_id = 0 if record.condition == "haze" else 1

        return {
            "template": template_tensor,
            "search": search_tensor,
            "template_restored": template_restored_tensor,
            "search_restored": search_restored_tensor,
            "target_box": target_tensor,
            "condition": torch.tensor(condition_id, dtype=torch.long),
        }
