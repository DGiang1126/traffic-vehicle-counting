"""Aggregate vehicle counting events into traffic statistics CSV files.

Input: one system ``*_events.csv`` file produced by the Counting stage.
Output: totals by class, minute, line, and direction when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "csv"
DEFAULT_CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
VEHICLE_CLASSES = ["motorcycle", "car", "bus", "truck", "bicycle"]
REQUIRED_COLUMNS = {"timestamp", "class"}


def load_events(path: Path) -> pd.DataFrame:
    """Load and validate a system counting-event CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Events CSV not found: {path}")

    events = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(
            "Events CSV is missing required column(s): " + ", ".join(sorted(missing))
        )
    if events.empty:
        return events

    timestamps = pd.to_numeric(events["timestamp"], errors="coerce")
    if timestamps.isna().any():
        bad_rows = events.index[timestamps.isna()].tolist()
        raise ValueError(f"Invalid timestamp at row(s): {bad_rows}")
    if (timestamps < 0).any():
        raise ValueError("timestamp must be >= 0")

    events = events.copy()
    events["timestamp"] = timestamps.astype(float)
    events["class"] = events["class"].astype(str).str.strip().str.lower()
    events = events[events["class"] != ""].reset_index(drop=True)
    unexpected = sorted(set(events["class"]) - set(VEHICLE_CLASSES))
    if unexpected:
        raise ValueError(
            "Events CSV contains class(es) outside the agreed vehicle set: "
            + ", ".join(unexpected)
        )
    return events


def ordered_classes(events: pd.DataFrame) -> list[str]:
    """Return project vehicle classes first, then any unexpected extra classes."""
    present = set(events["class"].dropna().astype(str)) if not events.empty else set()
    extras = sorted(present - set(VEHICLE_CLASSES))
    return [*VEHICLE_CLASSES, *extras]


def counts_by_class(events: pd.DataFrame) -> pd.DataFrame:
    classes = ordered_classes(events)
    counts = events["class"].value_counts() if not events.empty else pd.Series(dtype=int)
    rows = [{"class": cls, "count": int(counts.get(cls, 0))} for cls in classes]
    rows.append({"class": "TOTAL", "count": int(len(events))})
    return pd.DataFrame(rows, columns=["class", "count"])


def counts_by_interval(
    events: pd.DataFrame,
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """Aggregate counts into fixed time bins; 60 seconds gives minute format."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    classes = ordered_classes(events)
    columns = ["minute", "start_sec", "end_sec", *classes, "total"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    data = events.copy()
    data["time_bin"] = (data["timestamp"] // interval_seconds).astype(int)
    grouped = data.groupby(["time_bin", "class"]).size().unstack(fill_value=0)

    max_bin = int(data["time_bin"].max())
    grouped = grouped.reindex(range(max_bin + 1), fill_value=0)
    for cls in classes:
        if cls not in grouped.columns:
            grouped[cls] = 0
    grouped = grouped[classes]

    result = grouped.reset_index().rename(columns={"time_bin": "minute"})
    result.insert(1, "start_sec", result["minute"] * interval_seconds)
    result.insert(2, "end_sec", (result["minute"] + 1) * interval_seconds)
    result["total"] = result[classes].sum(axis=1).astype(int)
    return result[columns]


def counts_by_column(events: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate by an optional event column such as line or direction."""
    classes = ordered_classes(events)
    output_columns = [column, *classes, "total"]
    if column not in events.columns or events.empty:
        return pd.DataFrame(columns=output_columns)

    data = events.copy()
    data[column] = data[column].fillna("unknown").astype(str).str.strip()
    data.loc[data[column] == "", column] = "unknown"
    grouped = data.groupby([column, "class"]).size().unstack(fill_value=0)
    for cls in classes:
        if cls not in grouped.columns:
            grouped[cls] = 0
    grouped = grouped[classes]
    grouped["total"] = grouped.sum(axis=1).astype(int)
    return grouped.reset_index()[output_columns]


def build_statistics(events: pd.DataFrame, interval_seconds: int = 60) -> dict[str, pd.DataFrame]:
    """Build all statistics required by the project from event rows."""
    result = {
        "by_class": counts_by_class(events),
        "by_minute": counts_by_interval(events, interval_seconds),
    }
    if "line" in events.columns:
        result["by_line"] = counts_by_column(events, "line")
    if "direction" in events.columns:
        result["by_direction"] = counts_by_column(events, "direction")
    return result


def experiment_prefix(events_path: Path) -> str:
    stem = events_path.stem
    return stem[:-7] if stem.endswith("_events") else stem


def save_statistics(
    statistics: dict[str, pd.DataFrame],
    events_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Save summary CSV files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = experiment_prefix(events_path)
    paths: dict[str, Path] = {}

    naming = {
        "by_class": f"{prefix}_statistics.csv",
        "by_minute": f"{prefix}_statistics_by_minute.csv",
        "by_line": f"{prefix}_statistics_by_line.csv",
        "by_direction": f"{prefix}_statistics_by_direction.csv",
    }
    for key, frame in statistics.items():
        path = output_dir / naming[key]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    return paths



def plot_class_counts(by_class: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a bar chart of vehicle counts by class."""
    import matplotlib.pyplot as plt

    data = by_class[by_class["class"].str.upper() != "TOTAL"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(data["class"], data["count"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Vehicle class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, data["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(value)),
                ha="center", va="bottom")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_traffic_flow(by_time: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a line chart of total counted vehicles across time bins."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    if by_time.empty:
        ax.text(0.5, 0.5, "No counting events", ha="center", va="center", transform=ax.transAxes)
    else:
        x = by_time["start_sec"].astype(float)
        y = by_time["total"].astype(float)
        ax.plot(x, y, marker="o")
        ax.set_xticks(x)
        labels = [f"{float(a):g}-{float(b):g}s" for a, b in zip(by_time["start_sec"], by_time["end_sec"])]
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Vehicle count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_charts(
    statistics: dict[str, pd.DataFrame],
    events_path: Path,
    chart_dir: Path = DEFAULT_CHART_DIR,
) -> dict[str, Path]:
    """Create experiment-specific charts and task-compatible latest aliases."""
    import shutil

    chart_dir.mkdir(parents=True, exist_ok=True)
    prefix = experiment_prefix(events_path)
    class_path = chart_dir / f"{prefix}_class_counts.png"
    flow_path = chart_dir / f"{prefix}_traffic_flow.png"
    plot_class_counts(statistics["by_class"], class_path, f"Vehicle counts by class — {prefix}")
    plot_traffic_flow(statistics["by_minute"], flow_path, f"Traffic flow — {prefix}")

    latest_class = chart_dir / "class_counts.png"
    latest_flow = chart_dir / "traffic_flow.png"
    shutil.copyfile(class_path, latest_class)
    shutil.copyfile(flow_path, latest_flow)
    return {
        "class_counts": class_path,
        "traffic_flow": flow_path,
        "latest_class_counts": latest_class,
        "latest_traffic_flow": latest_flow,
    }

def run_statistics(
    events_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    interval_seconds: int = 60,
    chart_dir: Path = DEFAULT_CHART_DIR,
    create_charts: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    events = load_events(events_path)
    statistics = build_statistics(events, interval_seconds=interval_seconds)
    paths = save_statistics(statistics, events_path, output_dir)
    if create_charts:
        chart_paths = save_charts(statistics, events_path, chart_dir)
        paths.update({f"chart_{key}": value for key, value in chart_paths.items()})
    return statistics, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate vehicle counting events into statistics CSV files."
    )
    parser.add_argument("events_csv", type=Path, help="Path to a *_events.csv file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for statistics CSV files (default: outputs/csv)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Time-bin size in seconds (default: 60)",
    )
    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=DEFAULT_CHART_DIR,
        help="Directory for PNG charts (default: outputs/charts)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Generate CSV statistics only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics, paths = run_statistics(
        args.events_csv,
        output_dir=args.output_dir,
        interval_seconds=args.interval,
        chart_dir=args.chart_dir,
        create_charts=not args.no_charts,
    )

    total = int(statistics["by_class"].iloc[-1]["count"])
    print(f"Processed events: {total}")
    print("Saved statistics:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
