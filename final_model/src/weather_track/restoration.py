from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from .augmentations import build_condition_map, condition_vector, image_to_tensor, pad_to_square, tensor_to_image


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConditionedUNetGenerator(nn.Module):
    def __init__(self, condition_channels: int = 2, base_channels: int = 48) -> None:
        super().__init__()
        in_channels = 3 + condition_channels
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 8, base_channels * 4),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 8, base_channels * 2),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 4, base_channels),
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_channels * 2, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition_map = build_condition_map(condition, image.shape[-2], image.shape[-1])
        x = torch.cat([image, condition_map], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        return (self.out(d1) + 1.0) / 2.0


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 8, base_channels: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, condition_channels: int = 2) -> None:
        super().__init__()
        self.scale_1 = PatchDiscriminator(in_channels=6 + condition_channels)
        self.scale_2 = PatchDiscriminator(in_channels=6 + condition_channels)
        self.pool = nn.AvgPool2d(2)

    def forward(self, degraded: torch.Tensor, restored: torch.Tensor, condition: torch.Tensor) -> list[torch.Tensor]:
        condition_map = build_condition_map(condition, degraded.shape[-2], degraded.shape[-1])
        stacked = torch.cat([degraded, restored, condition_map], dim=1)
        low = self.pool(stacked)
        return [self.scale_1(stacked), self.scale_2(low)]


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            weights = VGG16_Weights.IMAGENET1K_V1
            features = vgg16(weights=weights).features[:16]
            self.enabled = True
        except Exception:
            features = nn.Identity()
            self.enabled = False
        self.features = features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return prediction.new_tensor(0.0)
        return F.l1_loss(self.features(prediction), self.features(target))


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = prediction - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=prediction.device)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=prediction.device)
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)

    def gradient_map(x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, kernel_x, padding=1)
        gy = F.conv2d(gray, kernel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    return F.l1_loss(gradient_map(prediction), gradient_map(target))


def adversarial_generator_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit)) for logit in fake_logits) / len(
        fake_logits
    )


def adversarial_discriminator_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    real_loss = sum(F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit)) for logit in real_logits)
    fake_loss = sum(F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit)) for logit in fake_logits)
    return (real_loss + fake_loss) / (len(real_logits) + len(fake_logits))


class ClassicalWeatherRestorer:
    def restore(self, frame: np.ndarray, weather_hint: str | None = None) -> np.ndarray:
        weather_hint = (weather_hint or "").upper()
        if weather_hint == "HAZY":
            return self._restore_haze(frame)
        if weather_hint == "RAIN":
            return self._restore_rain(frame)
        return self._restore_haze(frame)

    def _restore_haze(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge([l_channel, a_channel, b_channel])
        restored = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        restored = cv2.detailEnhance(restored, sigma_s=12, sigma_r=0.15)
        return restored

    def _restore_rain(self, frame: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 4, 4, 7, 21)
        smooth = cv2.bilateralFilter(denoised, 7, 40, 40)
        sharpened = cv2.addWeighted(denoised, 1.2, smooth, -0.2, 4)
        return sharpened


@dataclass
class RestorationInferenceEngine:
    image_size: int = 640
    checkpoint: Path | None = None
    device: str = "cpu"
    use_classical_fallback: bool = True

    def __post_init__(self) -> None:
        self.device_obj = torch.device(self.device)
        self.generator = ConditionedUNetGenerator().to(self.device_obj)
        self.classical = ClassicalWeatherRestorer()
        self.has_checkpoint = False
        if self.checkpoint and Path(self.checkpoint).exists():
            payload = torch.load(self.checkpoint, map_location=self.device_obj)
            state_dict = payload["generator"] if isinstance(payload, dict) and "generator" in payload else payload
            self.generator.load_state_dict(state_dict)
            self.generator.eval()
            self.has_checkpoint = True

    def restore(self, frame: np.ndarray, weather_hint: str | None = None) -> np.ndarray:
        if self.has_checkpoint:
            return self._restore_with_generator(frame, weather_hint)
        if self.use_classical_fallback:
            return self.classical.restore(frame, weather_hint)
        return frame.copy()

    def _restore_with_generator(self, frame: np.ndarray, weather_hint: str | None = None) -> np.ndarray:
        padded, _, _ = pad_to_square(frame, self.image_size)
        tensor = image_to_tensor(padded)[None, ...].to(self.device_obj)
        condition = condition_vector(weather_hint or "UNKNOWN", device=self.device_obj)[None, ...]
        with torch.inference_mode():
            restored = self.generator(tensor, condition)[0]
        restored_image = tensor_to_image(restored)
        return cv2.resize(restored_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
