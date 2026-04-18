from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


@dataclass
class DualPathDetections:
    original: Boxes
    restored: Boxes
    fused: Boxes


class YOLODetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cpu",
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        fusion_iou: float = 0.55,
        classes: list[int] | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.fusion_iou = fusion_iou
        self.classes = classes

    def detect(self, frame: np.ndarray) -> Boxes:
        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        return results[0].boxes.cpu()

    def dual_path_detect(self, original: np.ndarray, restored: np.ndarray) -> DualPathDetections:
        original_boxes = self.detect(original)
        restored_boxes = self.detect(restored)
        fused = self.fuse_boxes([original_boxes, restored_boxes], original.shape[:2])
        return DualPathDetections(original=original_boxes, restored=restored_boxes, fused=fused)

    def fuse_boxes(self, groups: list[Boxes], orig_shape: tuple[int, int]) -> Boxes:
        tensors: list[torch.Tensor] = []
        for group in groups:
            data = group.data
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if data.numel() > 0:
                tensors.append(data[:, :6].cpu())
        if not tensors:
            return Boxes(torch.empty((0, 6), dtype=torch.float32), orig_shape)

        stacked = torch.cat(tensors, dim=0)
        boxes_xyxy = stacked[:, :4]
        scores = stacked[:, 4]
        classes = stacked[:, 5]
        offsets = classes.unsqueeze(1) * 4096.0
        keep = nms(boxes_xyxy + torch.cat([offsets, offsets, offsets, offsets], dim=1), scores, self.fusion_iou)
        fused = stacked[keep]
        return Boxes(fused, orig_shape)
