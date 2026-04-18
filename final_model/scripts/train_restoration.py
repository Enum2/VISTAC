from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from weather_track.data import SyntheticWeatherDataset
from weather_track.restoration import (
    ConditionedUNetGenerator,
    MultiScaleDiscriminator,
    PerceptualLoss,
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    charbonnier_loss,
    edge_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the restoration GAN on synthetic haze/rain pairs.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SyntheticWeatherDataset(
        dataset_root=args.dataset_root,
        split="train",
        image_size=args.image_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    val_dataset = SyntheticWeatherDataset(
        dataset_root=args.dataset_root,
        split="val",
        image_size=args.image_size,
        samples_per_epoch=max(128, args.batch_size * 16),
        seed=args.seed + 101,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    generator = ConditionedUNetGenerator().to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    perceptual = PerceptualLoss().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        running_g = 0.0
        running_d = 0.0

        for batch in train_loader:
            clean = batch["clean"].to(device)
            degraded = batch["degraded"].to(device)
            condition = batch["condition"].to(device)

            restored = generator(degraded, condition)
            real_logits = discriminator(degraded, clean, condition)
            fake_logits = discriminator(degraded, restored.detach(), condition)
            d_loss = adversarial_discriminator_loss(real_logits, fake_logits)

            optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_d.step()

            restored = generator(degraded, condition)
            fake_logits = discriminator(degraded, restored, condition)
            g_adv = adversarial_generator_loss(fake_logits)
            g_rec = charbonnier_loss(restored, clean)
            g_edge = edge_loss(restored, clean)
            g_perc = perceptual(restored, clean)
            g_loss = g_adv + 10.0 * g_rec + 1.0 * g_edge + 0.1 * g_perc

            optimizer_g.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_g.step()

            running_g += float(g_loss.item())
            running_d += float(d_loss.item())

        generator.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                degraded = batch["degraded"].to(device)
                condition = batch["condition"].to(device)
                restored = generator(degraded, condition)
                val_loss += float(charbonnier_loss(restored, clean).item())
        val_loss /= max(1, len(val_loader))

        epoch_summary = {
            "epoch": float(epoch),
            "generator_loss": running_g / max(1, len(train_loader)),
            "discriminator_loss": running_d / max(1, len(train_loader)),
            "val_charbonnier": val_loss,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

        latest_path = args.output_dir / "restoration_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "history": history,
            },
            latest_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "history": history,
                    "best_val_charbonnier": best_val,
                },
                args.output_dir / "restoration_best.pt",
            )


if __name__ == "__main__":
    main()
