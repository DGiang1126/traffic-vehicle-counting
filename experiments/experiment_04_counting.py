"""E4: compare single-line and multi-line strategies without overwriting runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_counting import evaluate
from src.traffic_counting import run


def main():
    parser = argparse.ArgumentParser(description="E4 counting experiment")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "evaluation"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--single-config", default=str(PROJECT_ROOT / "configs" / "e4_single_line.json"))
    parser.add_argument("--multi-config", default=str(PROJECT_ROOT / "configs" / "e4_multi_line.json"))
    parser.add_argument("--single-ground-truth", default=None)
    parser.add_argument("--multi-ground-truth", default=None)
    args = parser.parse_args()

    video = Path(args.video)
    run_id = args.run_id or f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_root = Path(args.output_dir) / run_id
    results = []
    for strategy, config, ground_truth in (
        ("single_line", args.single_config, args.single_ground_truth),
        ("multi_line", args.multi_config, args.multi_ground_truth),
    ):
        run_name = f"{video.stem}_e4_{strategy}"
        result = run(video, config, run_root, run_name)
        row = {
            "strategy": strategy,
            "number_of_lines": result["number_of_lines"],
            "total_events": result["total_events"],
            "events_csv": result["events_csv"],
            "output_video": result["output_video"],
        }
        if ground_truth:
            evaluation_csv = run_root / f"{run_name}_evaluation.csv"
            row.update(evaluate(result["events_csv"], ground_truth, evaluation_csv))
        results.append(row)

    comparison_path = run_root / f"{video.stem}_e4_counting_comparison.json"
    comparison_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hoàn thành. Kết quả: {run_root}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
