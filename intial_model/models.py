from __future__ import annotations

import torch
from torch import nn


class ConvStem(nn.Module):
    def __init__(self, out_channels: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, raw: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([raw, restored], dim=1))


class MixformerBlock(nn.Module):
    def __init__(self, dim: int = 192, heads: int = 6, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.search_norm_1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.search_norm_2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.search_norm_3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, template_tokens: torch.Tensor, search_tokens: torch.Tensor) -> torch.Tensor:
        s = search_tokens + self.self_attn(
            self.search_norm_1(search_tokens),
            self.search_norm_1(search_tokens),
            self.search_norm_1(search_tokens),
            need_weights=False,
        )[0]
        s = s + self.cross_attn(
            self.search_norm_2(s),
            template_tokens,
            template_tokens,
            need_weights=False,
        )[0]
        s = s + self.mlp(self.search_norm_3(s))
        return s


class MixformerLiteTracker(nn.Module):
    def __init__(self, dim: int = 192, depth: int = 4, heads: int = 6, condition_dim: int = 16) -> None:
        super().__init__()
        self.template_stem = ConvStem(out_channels=dim)
        self.search_stem = ConvStem(out_channels=dim)
        self.blocks = nn.ModuleList([MixformerBlock(dim=dim, heads=heads) for _ in range(depth)])
        self.condition_embedding = nn.Embedding(2, condition_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim + condition_dim),
            nn.Linear(dim + condition_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 128),
            nn.GELU(),
        )
        self.box_head = nn.Linear(128, 4)
        self.score_head = nn.Linear(128, 1)

    def forward(
        self,
        template_raw: torch.Tensor,
        template_restored: torch.Tensor,
        search_raw: torch.Tensor,
        search_restored: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_feat = self.template_stem(template_raw, template_restored)
        s_feat = self.search_stem(search_raw, search_restored)

        t_tokens = t_feat.flatten(2).transpose(1, 2)
        s_tokens = s_feat.flatten(2).transpose(1, 2)
        for block in self.blocks:
            s_tokens = block(t_tokens, s_tokens)

        pooled = s_tokens.mean(dim=1)
        c_embed = self.condition_embedding(condition)
        fused = torch.cat([pooled, c_embed], dim=1)
        h = self.head(fused)
        box = torch.sigmoid(self.box_head(h))
        score_logits = self.score_head(h)
        return box, score_logits.squeeze(-1)


def build_model(name: str) -> nn.Module:
    if name in {"mixformer_lite", "tiny_siamese", "tiny_siamese_attn"}:
        # We keep old names mapped for backward compatibility.
        return MixformerLiteTracker(dim=192, depth=4, heads=6, condition_dim=16)
    if name == "mixformer_lite_large":
        return MixformerLiteTracker(dim=256, depth=6, heads=8, condition_dim=24)
    raise ValueError(f"Unknown model: {name}")
