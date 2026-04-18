from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RestorationConfig:
    checkpoint: Path | None = None
    device: str = field(default_factory=_default_device)
    image_size: int = 640
    use_classical_fallback: bool = True
    save_restored_frames: bool = False


@dataclass
class DetectionConfig:
    weights: str = "yolov8n.pt"
    device: str = field(default_factory=_default_device)
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    fusion_iou: float = 0.55
    classes: list[int] | None = None


@dataclass
class TrackingConfig:
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False
    reid_model: str = "auto"
    smoothing_alpha: float = 0.65


@dataclass
class OutputConfig:
    root: Path
    save_visualizations: bool = True
    save_restored_frames: bool = False
    save_json: bool = True
    save_csv: bool = True


@dataclass
class PipelineConfig:
    dataset_root: Path
    restoration: RestorationConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    output: OutputConfig


def build_default_config(dataset_root: Path) -> PipelineConfig:
    dataset_root = Path(dataset_root)
    outputs = dataset_root / "outputs"
    return PipelineConfig(
        dataset_root=dataset_root,
        restoration=RestorationConfig(),
        detection=DetectionConfig(),
        tracking=TrackingConfig(),
        output=OutputConfig(root=outputs),
    )
