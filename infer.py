from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vistac_tracker.dataset import load_sequences
from vistac_tracker.engine import evaluate_model
from vistac_tracker.models import build_model
from vistac_tracker.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference or validation for a trained ExtremeTrack tracker.")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/vistac/ExtremeTrack"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("D:/vistac/outputs/inference_predictions.json"))
    parser.add_argument("--use-restoration", action="store_true", default=True)
    parser.add_argument("--yolo-fallback", action="store_true", default=False)
    parser.add_argument("--yolo-model-name", type=str, default="yolov8n.pt")
    parser.add_argument("--iqa-cache-path", type=Path, default=Path("D:/vistac/outputs/cache/val_iqa_cache.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])

    if args.split == "val":
        sequences = load_sequences(args.dataset_root, "ExtremeTrack_val.json")
    else:
        raise ValueError(f"Unsupported split: {args.split}")

    metrics, predictions = evaluate_model(
        model,
        args.dataset_root,
        sequences,
        device,
        args.output_json,
        use_restoration=args.use_restoration,
        yolo_fallback=args.yolo_fallback,
        yolo_model_name=args.yolo_model_name,
        iqa_cache_path=args.iqa_cache_path,
    )
    save_json(args.output_json.with_suffix(".metrics.json"), metrics)
    print(metrics)
    print(f"Predictions saved to {args.output_json}")


if __name__ == "__main__":
    main()
