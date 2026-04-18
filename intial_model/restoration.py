from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _enhance_luminance(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def restore_frame(image: Image.Image, condition: str) -> Image.Image:
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    enhanced = _enhance_luminance(image_bgr)
    if condition == "rain":
        # Rain artifacts tend to be high-frequency streaks; median filter is cheap and effective.
        enhanced = cv2.medianBlur(enhanced, 3)
    else:
        # Haze often benefits from local contrast + edge-aware smoothing.
        enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)

