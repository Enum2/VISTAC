from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT


class BoTSORTTracker:
    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        gmc_method: str = "sparseOptFlow",
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.8,
        with_reid: bool = False,
        reid_model: str = "auto",
        smoothing_alpha: float = 0.65,
    ) -> None:
        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
            gmc_method=gmc_method,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            with_reid=with_reid,
            model=reid_model,
        )
        self.tracker = BOTSORT(args=args, frame_rate=30)
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_boxes: dict[int, np.ndarray] = {}

    def update(self, boxes: Boxes, frame: np.ndarray) -> list[dict[str, float | int]]:
        tracks = self.tracker.update(boxes.numpy(), frame, None)
        outputs: list[dict[str, float | int]] = []
        for track in tracks:
            x1, y1, x2, y2, track_id, score, cls_id, _det_idx = track.tolist()
            track_id = int(track_id)
            current_box = np.array([x1, y1, x2, y2], dtype=np.float32)
            if track_id in self.smoothed_boxes:
                current_box = self.smoothing_alpha * self.smoothed_boxes[track_id] + (1.0 - self.smoothing_alpha) * current_box
            self.smoothed_boxes[track_id] = current_box
            outputs.append(
                {
                    "track_id": track_id,
                    "class_id": int(cls_id),
                    "score": float(score),
                    "x1": float(current_box[0]),
                    "y1": float(current_box[1]),
                    "x2": float(current_box[2]),
                    "y2": float(current_box[3]),
                }
            )
        return outputs
