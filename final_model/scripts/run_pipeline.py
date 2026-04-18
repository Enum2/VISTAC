from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from weather_track.config import build_default_config
from weather_track.data import discover_sequences, get_sequence
from weather_track.pipeline import AdverseWeatherPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adverse-weather tracking pipeline.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=str, choices=["HAZY", "RAIN"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--restoration-checkpoint", type=Path, default=None)
    parser.add_argument("--detector-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--save-restored-frames", action="store_true")
    args = parser.parse_args()

    config = build_default_config(args.dataset_root)
    config.restoration.checkpoint = args.restoration_checkpoint
    config.restoration.save_restored_frames = args.save_restored_frames
    config.detection.weights = args.detector_weights
    config.output.save_visualizations = not args.no_visuals
    config.output.save_restored_frames = args.save_restored_frames

    pipeline = AdverseWeatherPipeline(config)

    if args.sequence:
        sequence = get_sequence(args.dataset_root, args.dataset, args.split, args.sequence)
    else:
        candidates = [
            seq
            for seq in discover_sequences(args.dataset_root)
            if seq.dataset == args.dataset.upper() and seq.split == args.split.lower()
        ]
        if not candidates:
            raise RuntimeError("No matching sequence found.")
        sequence = candidates[0]

    summary = pipeline.run_sequence(sequence, max_frames=args.max_frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
