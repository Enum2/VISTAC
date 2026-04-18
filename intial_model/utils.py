from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass
class CropInfo:
    left: float
    top: float
    side: float
    output_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def xywh_to_cxcywh(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x + (w / 2.0), y + (h / 2.0), w, h


def cxcywh_to_xywh(box: tuple[float, float, float, float]) -> list[float]:
    cx, cy, w, h = box
    return [cx - (w / 2.0), cy - (h / 2.0), w, h]


def clip_box_xywh(box: list[float], width: int, height: int, min_size: float = 2.0) -> list[float]:
    x, y, w, h = box
    w = max(min_size, min(w, width - 1))
    h = max(min_size, min(h, height - 1))
    x = min(max(0.0, x), max(0.0, width - w))
    y = min(max(0.0, y), max(0.0, height - h))
    return [float(x), float(y), float(w), float(h)]


def compute_square_crop(box: list[float], scale: float) -> tuple[float, float, float]:
    cx, cy, w, h = xywh_to_cxcywh(box)
    side = max(w, h) * scale
    side = max(side, 8.0)
    return cx, cy, side


def crop_square(image: Image.Image, cx: float, cy: float, side: float, out_size: int) -> tuple[Image.Image, CropInfo]:
    image_np = np.asarray(image)
    height, width = image_np.shape[:2]

    left = cx - (side / 2.0)
    top = cy - (side / 2.0)
    right = left + side
    bottom = top + side

    pad_left = max(0, int(math.ceil(-left)))
    pad_top = max(0, int(math.ceil(-top)))
    pad_right = max(0, int(math.ceil(right - width)))
    pad_bottom = max(0, int(math.ceil(bottom - height)))

    if pad_left or pad_top or pad_right or pad_bottom:
        image_np = np.pad(
            image_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        left += pad_left
        top += pad_top

    left_i = int(round(left))
    top_i = int(round(top))
    side_i = max(2, int(round(side)))
    crop = image_np[top_i : top_i + side_i, left_i : left_i + side_i]
    crop_img = Image.fromarray(crop).resize((out_size, out_size), Image.Resampling.BILINEAR)
    return crop_img, CropInfo(left=left_i - pad_left, top=top_i - pad_top, side=float(side_i), output_size=out_size)


def box_to_crop_normalized(box: list[float], crop: CropInfo) -> list[float]:
    x, y, w, h = box
    scale = crop.output_size / crop.side
    x = (x - crop.left) * scale
    y = (y - crop.top) * scale
    w = w * scale
    h = h * scale
    return [
        float((x + (w / 2.0)) / crop.output_size),
        float((y + (h / 2.0)) / crop.output_size),
        float(w / crop.output_size),
        float(h / crop.output_size),
    ]


def crop_normalized_to_box(box: list[float], crop: CropInfo, image_size: tuple[int, int]) -> list[float]:
    cx_n, cy_n, w_n, h_n = box
    crop_size = float(crop.output_size)
    cx = cx_n * crop_size
    cy = cy_n * crop_size
    w = max(2.0, w_n * crop_size)
    h = max(2.0, h_n * crop_size)
    scale = crop.side / crop.output_size
    x = crop.left + (cx - (w / 2.0)) * scale
    y = crop.top + (cy - (h / 2.0)) * scale
    w = w * scale
    h = h * scale
    return clip_box_xywh([x, y, w, h], width=image_size[0], height=image_size[1])


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = (aw * ah) + (bw * bh) - inter + 1e-6
    return float(inter / union)


def center_error(box_a: list[float], box_b: list[float]) -> float:
    acx, acy, _, _ = xywh_to_cxcywh(box_a)
    bcx, bcy, _, _ = xywh_to_cxcywh(box_b)
    return float(math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))

