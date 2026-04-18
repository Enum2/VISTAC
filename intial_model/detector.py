from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Detection:
    xywh: list[float]
    confidence: float


class YoloFallbackDetector:
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self.enabled = False
        self._model: Any = None
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_name)
            self.enabled = True
        except Exception:
            self.enabled = False

    def detect_best(self, image: Image.Image, reference_box: list[float]) -> Detection | None:
        if not self.enabled:
            return None
        try:
            result = self._model.predict(np.asarray(image), device=self.device, verbose=False, conf=0.15, imgsz=640)[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return None
            ref_cx = reference_box[0] + (reference_box[2] / 2.0)
            ref_cy = reference_box[1] + (reference_box[3] / 2.0)
            best = None
            best_score = -1e9
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = xyxy
                w = max(2.0, x2 - x1)
                h = max(2.0, y2 - y1)
                cx = x1 + (w / 2.0)
                cy = y1 + (h / 2.0)
                dist = ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
                score = conf - (0.0015 * dist)
                if score > best_score:
                    best_score = score
                    best = Detection(xywh=[x1, y1, w, h], confidence=conf)
            return best
        except Exception:
            return None

