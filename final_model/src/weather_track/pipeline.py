from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .data import SequenceInfo, iter_sequence_frames
from .detector import YOLODetector
from .metrics import DetectionAccumulator, RestorationAccumulator, TrackingAccumulator
from .restoration import RestorationInferenceEngine
from .tracker import BoTSORTTracker


def _annotate(frame: np.ndarray, tracks: list[dict[str, float | int]]) -> np.ndarray:
    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = int(track["x1"]), int(track["y1"]), int(track["x2"]), int(track["y2"])
        track_id = int(track["track_id"])
        score = float(track["score"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 120), 2)
        cv2.putText(
            canvas,
            f"id={track_id} conf={score:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


class AdverseWeatherPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.restorer = RestorationInferenceEngine(
            image_size=config.restoration.image_size,
            checkpoint=config.restoration.checkpoint,
            device=config.restoration.device,
            use_classical_fallback=config.restoration.use_classical_fallback,
        )
        self.detector = YOLODetector(
            weights=config.detection.weights,
            device=config.detection.device,
            image_size=config.detection.image_size,
            confidence=config.detection.confidence,
            iou=config.detection.iou,
            fusion_iou=config.detection.fusion_iou,
            classes=config.detection.classes,
        )
        self.tracker = BoTSORTTracker(
            track_high_thresh=config.tracking.track_high_thresh,
            track_low_thresh=config.tracking.track_low_thresh,
            new_track_thresh=config.tracking.new_track_thresh,
            track_buffer=config.tracking.track_buffer,
            match_thresh=config.tracking.match_thresh,
            fuse_score=config.tracking.fuse_score,
            gmc_method=config.tracking.gmc_method,
            proximity_thresh=config.tracking.proximity_thresh,
            appearance_thresh=config.tracking.appearance_thresh,
            with_reid=config.tracking.with_reid,
            reid_model=config.tracking.reid_model,
            smoothing_alpha=config.tracking.smoothing_alpha,
        )

    def run_sequence(self, sequence: SequenceInfo, max_frames: int | None = None) -> dict[str, object]:
        out_dir = self.config.output.root / sequence.dataset / sequence.split / sequence.name
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = out_dir / "visualizations"
        restored_dir = out_dir / "restored_frames"
        if self.config.output.save_visualizations:
            vis_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_restored_frames or self.config.restoration.save_restored_frames:
            restored_dir.mkdir(parents=True, exist_ok=True)

        restoration_metrics = RestorationAccumulator()
        detection_metrics = DetectionAccumulator()
        tracking_metrics = TrackingAccumulator()
        csv_rows: list[dict[str, object]] = []

        for frame_idx, frame_path in iter_sequence_frames(sequence, max_frames=max_frames):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            restored = self.restorer.restore(frame, weather_hint=sequence.weather)
            detections = self.detector.dual_path_detect(frame, restored)
            tracks = self.tracker.update(detections.fused, restored)

            restoration_metrics.update(frame, restored)
            detection_metrics.update(
                original_count=len(detections.original),
                restored_count=len(detections.restored),
                fused_count=len(detections.fused),
                fused_confidences=[float(conf) for conf in detections.fused.conf.tolist()] if len(detections.fused) else [],
            )
            tracking_metrics.update(tracks)

            for track in tracks:
                width = float(track["x2"]) - float(track["x1"])
                height = float(track["y2"]) - float(track["y1"])
                csv_rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": int(track["track_id"]),
                        "class_id": int(track["class_id"]),
                        "confidence": float(track["score"]),
                        "x": float(track["x1"]),
                        "y": float(track["y1"]),
                        "w": width,
                        "h": height,
                    }
                )

            if self.config.output.save_visualizations:
                annotated = _annotate(restored, tracks)
                cv2.imwrite(str(vis_dir / f"{frame_idx:05d}.jpg"), annotated)
            if self.config.output.save_restored_frames or self.config.restoration.save_restored_frames:
                cv2.imwrite(str(restored_dir / f"{frame_idx:05d}.jpg"), restored)

        summary = {
            "sequence": sequence.name,
            "dataset": sequence.dataset,
            "split": sequence.split,
            "weather": sequence.weather,
            "frames_processed": len(list(iter_sequence_frames(sequence, max_frames=max_frames))),
            "restoration": restoration_metrics.summary(),
            "detection": detection_metrics.summary(),
            "tracking": tracking_metrics.summary(),
            "expected_paper_metrics": {
                "avg_qp": 0.028,
                "avg_iou": 0.124,
                "success_rate_at_0_5": 0.031,
                "success_rate_at_20": 0.035,
                "avg_ote": 190.0,
            },
            "expected_ablation": {
                "yolov8_only": 0.015,
                "yolov8_plus_botsort": 0.021,
                "gan_plus_yolov8": 0.024,
                "full_model": 0.028,
            },
        }

        if self.config.output.save_csv:
            with (out_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["frame", "track_id", "class_id", "confidence", "x", "y", "w", "h"])
                writer.writeheader()
                writer.writerows(csv_rows)

        if self.config.output.save_json:
            with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

        with (out_dir / "summary.md").open("w", encoding="utf-8") as handle:
            handle.write(f"# Summary for {sequence.dataset}/{sequence.split}/{sequence.name}\n\n")
            handle.write("## Restoration\n")
            for key, value in summary["restoration"].items():
                handle.write(f"- {key}: {value:.4f}\n")
            handle.write("\n## Detection\n")
            for key, value in summary["detection"].items():
                handle.write(f"- {key}: {value:.4f}\n")
            handle.write("\n## Tracking\n")
            for key, value in summary["tracking"].items():
                handle.write(f"- {key}: {value:.4f}\n")
            handle.write("\n## Expected benchmark targets from the paper\n")
            for key, value in summary["expected_paper_metrics"].items():
                handle.write(f"- {key}: {value:.4f}\n")

        return summary
