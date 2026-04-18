from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
from typing import Iterable

import numpy as np
from brisque import BRISQUE
from PIL import Image

from .utils import bbox_iou, center_error


@dataclass
class SequenceMetrics:
    mean_iou: float
    precision_20: float
    success_auc: float
    mean_iqa: float
    qp: float


class IQAScorer:
    def __init__(self, cache_path: Path | None = None) -> None:
        self._brisque = BRISQUE(url=False)
        self.cache_path = cache_path
        self._cache: dict[str, float] = {}
        if self.cache_path is not None and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except Exception:
                self._cache = {}

    @lru_cache(maxsize=200000)
    def brisque_score(self, image_path: str) -> float:
        image = np.array(Image.open(Path(image_path)).convert("RGB"))
        return float(self._brisque.score(image))

    def iqa(self, image_path: str) -> float:
        if image_path in self._cache:
            return float(self._cache[image_path])
        brisque_score = self.brisque_score(image_path)
        value = float(np.clip(1.0 - (brisque_score / 100.0), 0.0, 1.0))
        self._cache[image_path] = value
        return value

    def warmup(self, image_paths: Iterable[str]) -> None:
        for path in image_paths:
            _ = self.iqa(path)

    def flush(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))


def evaluate_sequence(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    frame_iqas: list[float],
    qp_threshold: float = 15.0,
) -> SequenceMetrics:
    ious = np.array([bbox_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)], dtype=np.float32)
    center_errors = np.array([center_error(p, g) for p, g in zip(pred_boxes, gt_boxes)], dtype=np.float32)
    iqas = np.array(frame_iqas, dtype=np.float32)

    precision_20 = float((center_errors <= 20.0).mean())
    thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    success_curve = np.array([(ious >= t).mean() for t in thresholds], dtype=np.float32)
    success_auc = float(success_curve.mean())

    positive_frames = (iqas * center_errors) < qp_threshold
    qp = float(positive_frames.mean())
    return SequenceMetrics(
        mean_iou=float(ious.mean()),
        precision_20=precision_20,
        success_auc=success_auc,
        mean_iqa=float(iqas.mean()),
        qp=qp,
    )


def aggregate_metrics(sequence_metrics: list[SequenceMetrics]) -> dict[str, float]:
    keys = ["mean_iou", "precision_20", "success_auc", "mean_iqa", "qp"]
    return {
        key: float(np.mean([getattr(metric, key) for metric in sequence_metrics]))
        for key in keys
    }
