"""Export counting events, summaries and charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EVENT_COLUMNS = [
    "timestamp",
    "frame_index",
    "track_id",
    "class",
    "line",
    "direction",
    "confidence",
]


def events_dataframe(events) -> pd.DataFrame:
    """Return an event table with a stable schema, including when it is empty."""

    rows = [event.to_dict() if hasattr(event, "to_dict") else dict(event) for event in events]
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _write_summary(frame: pd.DataFrame, columns: list[str], output_path: Path) -> pd.DataFrame:
    if frame.empty:
        summary = pd.DataFrame(columns=[*columns, "count"])
    else:
        summary = frame.groupby(columns, dropna=False).size().reset_index(name="count")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def save_statistics(events, csv_path: Path, summary_dir: Path, chart_dir: Path) -> dict:
    """Save raw events, five summary CSV files, three charts and totals."""

    csv_path = Path(csv_path)
    summary_dir = Path(summary_dir)
    chart_dir = Path(chart_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    frame = events_dataframe(events)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summaries = {
        "by_class": _write_summary(frame, ["class"], summary_dir / "summary_by_class.csv"),
        "by_time": _write_summary(
            frame.assign(time_bin=(frame["timestamp"] // 60).astype(int) * 60)
            if not frame.empty
            else pd.DataFrame(columns=[*EVENT_COLUMNS, "time_bin"]),
            ["time_bin"],
            summary_dir / "summary_by_time.csv",
        ),
        "by_direction": _write_summary(
            frame, ["direction"], summary_dir / "summary_by_direction.csv"
        ),
        "by_line": _write_summary(frame, ["line"], summary_dir / "summary_by_line.csv"),
        "by_line_class_direction": _write_summary(
            frame,
            ["line", "class", "direction"],
            summary_dir / "summary_by_line_class_direction.csv",
        ),
    }

    chart_specs = [
        ("by_class", "class", "Số xe theo loại", "Loại xe", "count_by_class.png"),
        ("by_time", "time_bin", "Lưu lượng theo thời gian", "Giây", "traffic_flow_by_time.png"),
        ("by_line", "line", "Số xe theo counting line", "Counting line", "count_by_line.png"),
    ]
    chart_paths: dict[str, str] = {}
    for summary_key, x_column, title, xlabel, filename in chart_specs:
        summary = summaries[summary_key]
        path = chart_dir / filename
        plt.figure(figsize=(8, 4.5))
        if summary.empty:
            plt.text(0.5, 0.5, "Chưa có sự kiện đếm", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
        else:
            plt.bar(summary[x_column].astype(str), summary["count"])
            plt.xticks(rotation=20, ha="right")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Số lượng")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        chart_paths[summary_key] = str(path)

    totals = {
        "total_events": int(len(frame)),
        "events_csv": str(csv_path),
        "summary_dir": str(summary_dir),
        "chart_dir": str(chart_dir),
        "charts": chart_paths,
    }
    (summary_dir / "totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return totals
