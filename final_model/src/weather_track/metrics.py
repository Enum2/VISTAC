from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _entropy(gray_image: np.ndarray) -> float:
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    probabilities = histogram / max(1.0, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _laplacian_variance(image: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(image), cv2.CV_32F).var())


@dataclass
class RestorationAccumulator:
    contrast_gain: list[float] = field(default_factory=list)
    sharpness_gain: list[float] = field(default_factory=list)
    entropy_gain: list[float] = field(default_factory=list)

    def update(self, original: np.ndarray, restored: np.ndarray) -> None:
        original_gray = _gray(original)
        restored_gray = _gray(restored)
        self.contrast_gain.append(float(restored_gray.std() - original_gray.std()))
        self.sharpness_gain.append(_laplacian_variance(restored) - _laplacian_variance(original))
        self.entropy_gain.append(_entropy(restored_gray) - _entropy(original_gray))

    def summary(self) -> dict[str, float]:
        return {
            "mean_contrast_gain": float(np.mean(self.contrast_gain)) if self.contrast_gain else 0.0,
            "mean_sharpness_gain": float(np.mean(self.sharpness_gain)) if self.sharpness_gain else 0.0,
            "mean_entropy_gain": float(np.mean(self.entropy_gain)) if self.entropy_gain else 0.0,
        }


@dataclass
class DetectionAccumulator:
    original_counts: list[int] = field(default_factory=list)
    restored_counts: list[int] = field(default_factory=list)
    fused_counts: list[int] = field(default_factory=list)
    fused_confidences: list[float] = field(default_factory=list)

    def update(self, original_count: int, restored_count: int, fused_count: int, fused_confidences: list[float]) -> None:
        self.original_counts.append(original_count)
        self.restored_counts.append(restored_count)
        self.fused_counts.append(fused_count)
        self.fused_confidences.extend(fused_confidences)

    def summary(self) -> dict[str, float]:
        mean_conf = float(np.mean(self.fused_confidences)) if self.fused_confidences else 0.0
        return {
            "avg_original_detections": float(np.mean(self.original_counts)) if self.original_counts else 0.0,
            "avg_restored_detections": float(np.mean(self.restored_counts)) if self.restored_counts else 0.0,
            "avg_fused_detections": float(np.mean(self.fused_counts)) if self.fused_counts else 0.0,
            "mean_fused_confidence": mean_conf,
            "fusion_gain_vs_best_single": (
                float(np.mean(self.fused_counts) - max(np.mean(self.original_counts), np.mean(self.restored_counts)))
                if self.fused_counts
                else 0.0
            ),
        }


@dataclass
class TrackingAccumulator:
    track_frames: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    track_scores: list[float] = field(default_factory=list)
    step_motion: list[float] = field(default_factory=list)
    last_centers: dict[int, tuple[float, float]] = field(default_factory=dict)

    def update(self, tracks: list[dict[str, float | int]]) -> None:
        for track in tracks:
            track_id = int(track["track_id"])
            self.track_frames[track_id] += 1
            self.track_scores.append(float(track["score"]))
            center = ((float(track["x1"]) + float(track["x2"])) / 2.0, (float(track["y1"]) + float(track["y2"])) / 2.0)
            if track_id in self.last_centers:
                prev = self.last_centers[track_id]
                self.step_motion.append(float(np.hypot(center[0] - prev[0], center[1] - prev[1])))
            self.last_centers[track_id] = center

    def summary(self) -> dict[str, float]:
        lengths = list(self.track_frames.values())
        return {
            "unique_tracks": float(len(lengths)),
            "avg_track_length": float(np.mean(lengths)) if lengths else 0.0,
            "longest_track_length": float(np.max(lengths)) if lengths else 0.0,
            "mean_track_confidence": float(np.mean(self.track_scores)) if self.track_scores else 0.0,
            "mean_track_motion": float(np.mean(self.step_motion)) if self.step_motion else 0.0,
        }
