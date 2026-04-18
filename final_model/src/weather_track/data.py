from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import apply_weather, condition_vector, image_to_tensor, random_crop_resize, random_flip


def _frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem), stem
    except ValueError:
        return 10**9, stem


def _resolve_split_dir(dataset_dir: Path, token: str) -> Path:
    matches = [child for child in dataset_dir.iterdir() if child.is_dir() and token.lower() in child.name.lower()]
    if not matches:
        raise FileNotFoundError(f"Could not find a '{token}' directory inside {dataset_dir}")
    return matches[0]


@dataclass(frozen=True)
class SequenceInfo:
    dataset: str
    split: str
    weather: str
    name: str
    frame_dir: Path
    frame_paths: tuple[Path, ...]

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)


def discover_sequences(dataset_root: Path) -> list[SequenceInfo]:
    dataset_root = Path(dataset_root)
    sequences: list[SequenceInfo] = []
    for weather in ("HAZY", "RAIN"):
        weather_dir = dataset_root / weather
        if not weather_dir.exists():
            continue
        for split_name, token in (("train", "train"), ("val", "val")):
            split_dir = _resolve_split_dir(weather_dir, token)
            for sequence_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
                frames = tuple(sorted(sequence_dir.glob("*.jpg"), key=_frame_sort_key))
                if not frames:
                    continue
                sequences.append(
                    SequenceInfo(
                        dataset=weather,
                        split=split_name,
                        weather=weather,
                        name=sequence_dir.name,
                        frame_dir=sequence_dir,
                        frame_paths=frames,
                    )
                )
    return sequences


def get_sequence(dataset_root: Path, dataset: str, split: str, name: str) -> SequenceInfo:
    dataset = dataset.upper()
    split = split.lower()
    for sequence in discover_sequences(dataset_root):
        if sequence.dataset == dataset and sequence.split == split and sequence.name == name:
            return sequence
    raise FileNotFoundError(f"Sequence not found: dataset={dataset} split={split} name={name}")


def iter_sequence_frames(sequence: SequenceInfo, max_frames: int | None = None) -> Iterator[tuple[int, Path]]:
    frame_paths = sequence.frame_paths[:max_frames] if max_frames else sequence.frame_paths
    for index, frame_path in enumerate(frame_paths, start=1):
        yield index, frame_path


class SyntheticWeatherDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        dataset_root: Path,
        split: str = "train",
        image_size: int = 256,
        samples_per_epoch: int = 4096,
        max_frames_per_sequence: int = 40,
        seed: int = 13,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split.lower()
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.max_frames_per_sequence = max_frames_per_sequence
        self.seed = seed
        self.samples: list[tuple[Path, str]] = []

        for sequence in discover_sequences(self.dataset_root):
            if sequence.split != self.split:
                continue
            limited_frames = sequence.frame_paths[: self.max_frames_per_sequence]
            self.samples.extend((frame_path, sequence.weather) for frame_path in limited_frames)

        if not self.samples:
            raise RuntimeError(f"No samples found for split='{self.split}' under {self.dataset_root}")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        frame_path, weather = self.samples[index % len(self.samples)]
        rng = np.random.default_rng(self.seed + index)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {frame_path}")

        clean = random_crop_resize(image, self.image_size, rng)
        clean = random_flip(clean, rng)
        degraded = apply_weather(clean, weather, rng)

        return {
            "clean": image_to_tensor(clean),
            "degraded": image_to_tensor(degraded),
            "condition": condition_vector(weather),
            "weather": weather,
            "path": str(frame_path),
        }
