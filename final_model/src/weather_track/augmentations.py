from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
import torch


def resize_with_aspect(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) == size:
        return image.copy()
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_square(image: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    resized = resize_with_aspect(image, size)
    h, w = resized.shape[:2]
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top = (size - h) // 2
    left = (size - w) // 2
    canvas[top : top + h, left : left + w] = resized
    scale = h / image.shape[0]
    return canvas, scale, (left, top)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    return tensor.permute(2, 0, 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    rgb = (array * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_synthetic_haze(image: np.ndarray, rng: np.random.Generator, severity: float | None = None) -> np.ndarray:
    severity = float(severity if severity is not None else rng.uniform(0.35, 0.75))
    height, width = image.shape[:2]
    depth_gradient = np.linspace(0.1, 1.0, width, dtype=np.float32)
    depth_map = np.tile(depth_gradient[None, :], (height, 1))
    depth_map = cv2.GaussianBlur(depth_map, (0, 0), sigmaX=width / 18.0, sigmaY=height / 18.0)
    transmission = np.exp(-severity * depth_map)[..., None]
    atmospheric_light = np.full_like(image, int(255 * rng.uniform(0.78, 0.95)), dtype=np.float32)
    hazy = image.astype(np.float32) * transmission + atmospheric_light * (1.0 - transmission)
    hazy = cv2.GaussianBlur(hazy, (0, 0), sigmaX=max(0.6, severity * 1.6))
    return np.clip(hazy, 0, 255).astype(np.uint8)


def add_synthetic_rain(image: np.ndarray, rng: np.random.Generator, density: float | None = None) -> np.ndarray:
    density = float(density if density is not None else rng.uniform(0.08, 0.22))
    height, width = image.shape[:2]
    rain_layer = np.zeros((height, width), dtype=np.float32)
    streak_count = max(250, int(height * width * density / 1200))
    angle = rng.uniform(-25.0, 25.0)
    rad = math.radians(angle)
    line_dx = int(math.cos(rad) * rng.uniform(8, 18))
    line_dy = int(math.sin(rad) * rng.uniform(8, 18) + rng.uniform(12, 22))
    for _ in range(streak_count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        thickness = int(rng.integers(1, 2))
        color = float(rng.uniform(0.55, 0.92))
        cv2.line(rain_layer, (x, y), (x + line_dx, y + line_dy), color, thickness)
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_rgb = np.repeat(rain_layer[..., None], 3, axis=2)
    base = image.astype(np.float32) / 255.0
    blended = np.clip(base * (1.0 - density * 0.7) + rain_rgb * 0.8, 0.0, 1.0)
    blended = cv2.GaussianBlur(blended, (0, 0), sigmaX=0.5)
    return (blended * 255.0).astype(np.uint8)


def random_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        return cv2.flip(image, 1)
    return image


def random_crop_resize(image: np.ndarray, out_size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    if min(h, w) < 64:
        return cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    crop_scale = rng.uniform(0.72, 1.0)
    crop_h = max(32, int(h * crop_scale))
    crop_w = max(32, int(w * crop_scale))
    y1 = int(rng.integers(0, max(1, h - crop_h + 1)))
    x1 = int(rng.integers(0, max(1, w - crop_w + 1)))
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def apply_weather(image: np.ndarray, weather: str, rng: np.random.Generator) -> np.ndarray:
    if weather.upper() == "HAZY":
        return add_synthetic_haze(image, rng)
    if weather.upper() == "RAIN":
        return add_synthetic_rain(image, rng)
    if rng.random() < 0.5:
        return add_synthetic_haze(image, rng)
    return add_synthetic_rain(image, rng)


def condition_vector(weather: str, device: torch.device | str | None = None) -> torch.Tensor:
    if weather.upper() == "HAZY":
        tensor = torch.tensor([1.0, 0.0], dtype=torch.float32)
    elif weather.upper() == "RAIN":
        tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
    else:
        tensor = torch.tensor([0.5, 0.5], dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def build_condition_map(condition: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if condition.ndim == 1:
        condition = condition[None, :]
    return condition[:, :, None, None].expand(condition.shape[0], condition.shape[1], height, width)
