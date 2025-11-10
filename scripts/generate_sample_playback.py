"""
Generate a JSON Lines playback file using DEAP filtered labels.

Usage:
    python scripts/generate_sample_playback.py --output recordings/sample_deap_session.jsonl \
        --count 120 --step 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import csv
from typing import Iterable, List, Tuple


def load_labels(base_path: Path) -> Tuple[List[float], List[float]]:
    csv_path = base_path / "dataset_valence_top30.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "dataset_valence_top30.csv not found under "
            f"{base_path}. Ensure the filtered features notebook has been run."
        )

    valence_vals = []
    arousal_vals = []
    with csv_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                valence_vals.append(float(row["valence_continuous"]))
                arousal_vals.append(float(row["arousal_continuous"]))
            except (KeyError, ValueError):
                continue

    if not valence_vals:
        raise ValueError(f"No valence_continuous values found in {csv_path}")

    return valence_vals, arousal_vals


def iter_records(
    valence: List[float],
    arousal: List[float],
    count: int,
    step: float,
    start_time: float,
) -> Iterable[dict]:
    for idx in range(count):
        v = valence[idx % len(valence)]
        a = arousal[idx % len(arousal)]
        yield {
            "timestamp": start_time + idx * step,
            "valence_original": v,
            "arousal_original": a,
            "valence": v,
            "arousal": a,
            "valence_binary": int(v > 5.0),
            "arousal_binary": int(a > 5.0),
            "meta": {
                "source": "DEAP_filtered_features",
                "scale": "deap_1-9",
                "scale_max": 9.0,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DEAP playback session file.")
    parser.add_argument(
        "--base-path",
        default="EEG Prediction/datasets/DEAP/filtered_features",
        help="Directory containing y_valence_continuous.npy and y_arousal_continuous.npy",
    )
    parser.add_argument(
        "--output",
        default="recordings/sample_deap_session.jsonl",
        help="Path to the output JSONL file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=120,
        help="Number of records to emit (default corresponds to 1 minute at 0.5s step)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.5,
        help="Time step between records in seconds (playback respects original timing)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    valence, arousal = load_labels(base_path)
    start_time = time.time()

    with output.open("w", encoding="utf-8") as fh:
        for record in iter_records(valence, arousal, args.count, args.step, start_time):
            fh.write(json.dumps(record) + "\n")

    print(f"Wrote {args.count} records to {output}")


if __name__ == "__main__":
    main()

