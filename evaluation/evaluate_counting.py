"""Evaluate system vehicle counts against manual Ground Truth.

Inputs:
- Manual GT CSV owned by Evaluation (minute or start/end interval format).
- System statistics CSV produced by src/statistics/statistics.py.
- Optional system statistics-by-time CSV for temporal MAE.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "csv"
VEHICLE_CLASSES = ["motorcycle", "car", "bus", "truck", "bicycle"]
TIME_TOLERANCE = 1e-3


def _numeric_nonnegative(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = 0
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any():
            rows = result.index[values.isna()].tolist()
            raise ValueError(f"{label}: invalid {column} value at row(s) {rows}")
        if (values < 0).any():
            raise ValueError(f"{label}: {column} must be >= 0")
        result[column] = values.astype(float)
    return result


def load_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground Truth CSV not found: {path}")
    gt = pd.read_csv(path)
    if gt.empty:
        raise ValueError("Ground Truth CSV is empty")

    has_minute = "minute" in gt.columns
    has_range = {"start_sec", "end_sec"}.issubset(gt.columns)
    if not has_minute and not has_range:
        raise ValueError(
            "Ground Truth must contain either 'minute' or both 'start_sec' and 'end_sec'"
        )

    gt = _numeric_nonnegative(gt, VEHICLE_CLASSES, "Ground Truth")
    if has_minute:
        minute = pd.to_numeric(gt["minute"], errors="coerce")
        if minute.isna().any() or (minute < 0).any():
            raise ValueError("Ground Truth: minute must be a non-negative number")
        gt["minute"] = minute.astype(int)
    if has_range:
        for col in ("start_sec", "end_sec"):
            gt[col] = pd.to_numeric(gt[col], errors="coerce")
        if gt[["start_sec", "end_sec"]].isna().any().any():
            raise ValueError("Ground Truth: start_sec/end_sec must be numeric")
        if (gt["start_sec"] < 0).any() or (gt["end_sec"] <= gt["start_sec"]).any():
            raise ValueError("Ground Truth: invalid start_sec/end_sec interval")
    return gt


def load_system_by_class(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"System statistics CSV not found: {path}")
    stats = pd.read_csv(path)
    required = {"class", "count"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError("System statistics missing: " + ", ".join(sorted(missing)))
    stats = stats.copy()
    stats["class"] = stats["class"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(stats["class"]) - set(VEHICLE_CLASSES) - {"total"})
    if unexpected:
        raise ValueError(
            "System statistics contains class(es) outside the agreed vehicle set: "
            + ", ".join(unexpected)
        )
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce")
    if stats["count"].isna().any() or (stats["count"] < 0).any():
        raise ValueError("System statistics: count must be numeric and >= 0")
    return stats


def load_system_by_time(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"System time statistics CSV not found: {path}")
    stats = pd.read_csv(path)
    has_minute = "minute" in stats.columns
    has_range = {"start_sec", "end_sec"}.issubset(stats.columns)
    if not has_minute and not has_range:
        raise ValueError(
            "System time statistics must contain 'minute' or start_sec/end_sec"
        )
    stats = _numeric_nonnegative(stats, VEHICLE_CLASSES, "System time statistics")
    if has_minute:
        stats["minute"] = pd.to_numeric(stats["minute"], errors="coerce")
        if stats["minute"].isna().any() or (stats["minute"] < 0).any():
            raise ValueError("System time statistics: minute must be >= 0")
        stats["minute"] = stats["minute"].astype(int)
    if has_range:
        stats["start_sec"] = pd.to_numeric(stats["start_sec"], errors="coerce")
        stats["end_sec"] = pd.to_numeric(stats["end_sec"], errors="coerce")
        if stats[["start_sec", "end_sec"]].isna().any().any():
            raise ValueError("System time statistics: invalid start/end time")
    return stats


def counting_accuracy(manual: float, system: float) -> float:
    """Project counting-accuracy formula in percent, with zero-GT handling."""
    if manual == 0:
        return 100.0 if system == 0 else 0.0
    return 100.0 * (1.0 - abs(system - manual) / manual)


def metric_row(label: str, manual: float, system: float) -> dict[str, object]:
    signed = system - manual
    return {
        "class": label,
        "manual_count": int(round(manual)),
        "system_count": int(round(system)),
        "signed_error": int(round(signed)),
        "absolute_error": int(round(abs(signed))),
        "error_type": "overcount" if signed > 0 else "undercount" if signed < 0 else "exact",
        "accuracy_pct": round(counting_accuracy(manual, system), 4),
    }


def evaluate_by_class(gt: pd.DataFrame, system: pd.DataFrame) -> pd.DataFrame:
    manual_counts = gt[VEHICLE_CLASSES].sum(axis=0)
    system_rows = system[system["class"] != "total"]
    system_counts = system_rows.groupby("class")["count"].sum()

    rows = [
        metric_row(cls, float(manual_counts.get(cls, 0)), float(system_counts.get(cls, 0)))
        for cls in VEHICLE_CLASSES
    ]
    manual_total = float(manual_counts.sum())
    system_total = float(sum(system_counts.get(cls, 0) for cls in VEHICLE_CLASSES))
    rows.append(metric_row("TOTAL", manual_total, system_total))
    return pd.DataFrame(rows)


def _time_keyed(frame: pd.DataFrame, prefer_range: bool) -> pd.DataFrame:
    result = frame.copy()
    if prefer_range:
        if "start_sec" not in result.columns:
            raise ValueError("Time-bin mismatch: start_sec is required to match interval Ground Truth")
        result["time_key"] = result["start_sec"].round(3)
        result["time_label"] = result.apply(
            lambda r: f"{r['start_sec']:.3f}-{r['end_sec']:.3f}s", axis=1
        )
    else:
        if "minute" not in result.columns:
            raise ValueError("Time-bin mismatch: minute column is required for minute Ground Truth")
        result["time_key"] = result["minute"].astype(int)
        result["time_label"] = result["minute"].map(lambda value: f"minute_{value}")
    return result


def evaluate_by_time(
    gt: pd.DataFrame,
    system_time: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefer_range = {"start_sec", "end_sec"}.issubset(gt.columns)
    manual = _time_keyed(gt, prefer_range)
    system = _time_keyed(system_time, prefer_range)

    manual_keys = set(manual["time_key"].tolist())
    system_keys = set(system["time_key"].tolist())
    extra = sorted(system_keys - manual_keys)
    if extra:
        raise ValueError(
            "System Statistics contains time bins outside Ground Truth: "
            f"{extra}. Check that both sides use the same evaluation time range."
        )

    if prefer_range and not system.empty and len(manual_keys) > 1:
        last_manual_key = max(manual_keys)
        for key in sorted(manual_keys & system_keys):
            if key == last_manual_key:
                continue
            manual_end = float(manual.loc[manual["time_key"] == key, "end_sec"].iloc[0])
            system_end = float(system.loc[system["time_key"] == key, "end_sec"].iloc[0])
            if abs(manual_end - system_end) > TIME_TOLERANCE:
                raise ValueError(
                    "Time-bin size mismatch between Ground Truth and System Statistics. "
                    "Rerun statistics.py with the same interval used by Ground Truth."
                )

    manual = manual.set_index("time_key").sort_index()
    system = system.set_index("time_key").sort_index().reindex(manual.index)
    for cls in VEHICLE_CLASSES:
        system[cls] = system[cls].fillna(0)

    time_rows: list[dict[str, object]] = []
    class_time_rows: list[dict[str, object]] = []
    for key in manual.index:
        manual_row = manual.loc[key]
        system_row = system.loc[key]
        manual_total = float(sum(manual_row.get(cls, 0) for cls in VEHICLE_CLASSES))
        system_total = float(sum(system_row.get(cls, 0) for cls in VEHICLE_CLASSES))
        total_metrics = metric_row(str(manual_row["time_label"]), manual_total, system_total)
        time_rows.append({
            "time_bin": total_metrics.pop("class"),
            **total_metrics,
        })

        for cls in VEHICLE_CLASSES:
            metrics = metric_row(
                cls,
                float(manual_row.get(cls, 0)),
                float(system_row.get(cls, 0)),
            )
            class_time_rows.append({
                "time_bin": str(manual_row["time_label"]),
                **metrics,
            })

    return pd.DataFrame(time_rows), pd.DataFrame(class_time_rows)


def build_summary(
    experiment: str,
    by_class: pd.DataFrame,
    by_time: pd.DataFrame | None,
) -> pd.DataFrame:
    total = by_class[by_class["class"] == "TOTAL"].iloc[0]
    class_rows = by_class[by_class["class"] != "TOTAL"]
    present_classes = class_rows[class_rows["manual_count"] > 0]
    macro_accuracy = (
        float(present_classes["accuracy_pct"].mean())
        if not present_classes.empty
        else 100.0
    )
    summary = {
        "experiment": experiment,
        "manual_total": int(total["manual_count"]),
        "system_total": int(total["system_count"]),
        "signed_error": int(total["signed_error"]),
        "absolute_error": int(total["absolute_error"]),
        "counting_accuracy_pct": float(total["accuracy_pct"]),
        "mean_class_absolute_error": round(float(class_rows["absolute_error"].mean()), 4),
        "macro_class_accuracy_pct": round(macro_accuracy, 4),
        "time_mae": math.nan,
    }
    if by_time is not None and not by_time.empty:
        summary["time_mae"] = round(float(by_time["absolute_error"].mean()), 4)
    return pd.DataFrame([summary])


def infer_time_statistics_path(system_statistics: Path) -> Path:
    stem = system_statistics.stem
    if stem.endswith("_statistics"):
        return system_statistics.with_name(stem + "_by_minute.csv")
    return system_statistics.with_name(stem + "_by_minute.csv")


def experiment_name(system_statistics: Path, requested: str | None) -> str:
    if requested:
        return requested.strip()
    stem = system_statistics.stem
    return stem[:-11] if stem.endswith("_statistics") else stem


def save_evaluation(
    experiment: str,
    output_dir: Path,
    summary: pd.DataFrame,
    by_class: pd.DataFrame,
    by_time: pd.DataFrame | None,
    by_time_class: pd.DataFrame | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "summary": summary,
        "by_class": by_class,
    }
    if by_time is not None:
        frames["by_time"] = by_time
    if by_time_class is not None:
        frames["by_time_class"] = by_time_class

    paths: dict[str, Path] = {}
    for key, frame in frames.items():
        path = output_dir / f"{experiment}_evaluation_{key}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    return paths


def run_evaluation(
    ground_truth_path: Path,
    system_statistics_path: Path,
    system_time_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiment: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    gt = load_ground_truth(ground_truth_path)
    system = load_system_by_class(system_statistics_path)
    name = experiment_name(system_statistics_path, experiment)
    by_class = evaluate_by_class(gt, system)

    chosen_time_path = system_time_path or infer_time_statistics_path(system_statistics_path)
    by_time = None
    by_time_class = None
    if chosen_time_path.exists():
        system_time = load_system_by_time(chosen_time_path)
        by_time, by_time_class = evaluate_by_time(gt, system_time)
    elif system_time_path is not None:
        raise FileNotFoundError(f"System time statistics CSV not found: {chosen_time_path}")

    summary = build_summary(name, by_class, by_time)
    frames = {"summary": summary, "by_class": by_class}
    if by_time is not None:
        frames["by_time"] = by_time
        frames["by_time_class"] = by_time_class
    paths = save_evaluation(name, output_dir, summary, by_class, by_time, by_time_class)
    return frames, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare system counting statistics against manual Ground Truth."
    )
    parser.add_argument("ground_truth_csv", type=Path, help="Manual Ground Truth CSV")
    parser.add_argument("system_statistics_csv", type=Path, help="System *_statistics.csv")
    parser.add_argument(
        "--system-by-time",
        type=Path,
        help="Optional system statistics-by-time CSV; inferred automatically when omitted",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment label, e.g. e1_baseline; inferred from filename when omitted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation CSV files (default: outputs/csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, paths = run_evaluation(
        args.ground_truth_csv,
        args.system_statistics_csv,
        system_time_path=args.system_by_time,
        output_dir=args.output_dir,
        experiment=args.experiment,
    )

    summary = frames["summary"].iloc[0]
    print(f"Experiment: {summary['experiment']}")
    print(
        "Overall: "
        f"manual={int(summary['manual_total'])} | "
        f"system={int(summary['system_total'])} | "
        f"AE={int(summary['absolute_error'])} | "
        f"accuracy={summary['counting_accuracy_pct']:.2f}%"
    )
    if not pd.isna(summary["time_mae"]):
        print(f"Time MAE: {summary['time_mae']:.4f}")
    else:
        print("Time MAE: not computed (matching system time statistics not found)")
    print("Saved evaluation:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
