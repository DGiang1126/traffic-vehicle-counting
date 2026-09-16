"""Compare system event CSV with manually annotated ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def evaluate(prediction_csv: str | Path, ground_truth_csv: str | Path, output_csv=None) -> dict:
    prediction = pd.read_csv(prediction_csv)
    ground_truth = pd.read_csv(ground_truth_csv)
    required = {"class", "direction"}
    for name, table in (("prediction", prediction), ("ground truth", ground_truth)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} thiếu cột: {sorted(missing)}")

    keys = ["class", "direction"]
    if "line" in prediction.columns and "line" in ground_truth.columns:
        keys.insert(0, "line")
    pred_counts = prediction.groupby(keys).size().reset_index(name="predicted")
    gt_counts = ground_truth.groupby(keys).size().reset_index(name="ground_truth")
    comparison = gt_counts.merge(pred_counts, on=keys, how="outer").fillna(0)
    comparison[["ground_truth", "predicted"]] = comparison[["ground_truth", "predicted"]].astype(int)
    comparison["absolute_error"] = (comparison["predicted"] - comparison["ground_truth"]).abs()

    gt_total = int(comparison["ground_truth"].sum())
    predicted_total = int(comparison["predicted"].sum())
    absolute_error = abs(predicted_total - gt_total)
    metrics = {
        "ground_truth_total": gt_total,
        "predicted_total": predicted_total,
        "absolute_error": absolute_error,
        "over_count": max(predicted_total - gt_total, 0),
        "under_count": max(gt_total - predicted_total, 0),
        "accuracy": max(0.0, 1.0 - absolute_error / gt_total) if gt_total else float(predicted_total == 0),
        "mae_by_group": float(comparison["absolute_error"].mean()) if len(comparison) else 0.0,
    }
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(evaluate(args.prediction, args.ground_truth, args.output))


if __name__ == "__main__":
    main()

