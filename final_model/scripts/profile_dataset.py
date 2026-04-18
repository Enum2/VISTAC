from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from weather_track.data import discover_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the HAZY and RAIN datasets in the current workspace.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"sequences": 0, "frames": 0})
    for sequence in discover_sequences(args.dataset_root):
        key = (sequence.dataset, sequence.split)
        grouped[key]["sequences"] += 1
        grouped[key]["frames"] += sequence.num_frames

    serializable = {f"{dataset}/{split}": stats for (dataset, split), stats in grouped.items()}
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
