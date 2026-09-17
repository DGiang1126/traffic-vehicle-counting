"""Build the final cross-experiment results table.

Input: a manifest CSV describing each experiment and pointing to its
``*_evaluation_summary.csv`` file.
Output: one ``evaluation/results_summary.csv`` master table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results_summary.csv"
REQUIRED_MANIFEST_COLUMNS = {
    "experiment", "detector", "tracker", "counting", "evaluation_summary"
}
REQUIRED_SUMMARY_COLUMNS = {
    "experiment", "manual_total", "system_total", "signed_error",
    "absolute_error", "counting_accuracy_pct", "mean_class_absolute_error",
    "macro_class_accuracy_pct", "time_mae",
}
NUMERIC_SUMMARY_COLUMNS = [
    "manual_total", "system_total", "signed_error", "absolute_error",
    "counting_accuracy_pct", "mean_class_absolute_error",
    "macro_class_accuracy_pct", "time_mae",
]


def portable_path(path: Path) -> str:
    """Return a portable project-relative path for CSV artifacts."""
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_summary_path(raw: str, manifest_path: Path) -> Path:
    """Resolve summary paths portably from project root or manifest directory."""
    path = Path(raw.strip())
    if path.is_absolute():
        return path
    project_candidate = PROJECT_ROOT / path
    manifest_candidate = manifest_path.parent / path
    if project_candidate.exists():
        return project_candidate
    if manifest_candidate.exists():
        return manifest_candidate
    project_roots = {"outputs", "evaluation", "data", "src", "docs", "experiments", "models", "tests"}
    if path.parts and path.parts[0].lower() in project_roots:
        return project_candidate
    return manifest_candidate


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    if manifest.empty:
        raise ValueError("Manifest CSV is empty")
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError("Manifest missing required column(s): " + ", ".join(sorted(missing)))
    for column in REQUIRED_MANIFEST_COLUMNS:
        manifest[column] = manifest[column].astype(str).str.strip()
        if (manifest[column] == "").any():
            rows = manifest.index[manifest[column] == ""].tolist()
            raise ValueError(f"Manifest: '{column}' is empty at row(s) {rows}")

    duplicates = manifest[manifest["experiment"].duplicated(keep=False)]["experiment"].tolist()
    if duplicates:
        raise ValueError("Manifest contains duplicate experiment(s): " + ", ".join(sorted(set(duplicates))))

    if "fps" in manifest.columns:
        fps_raw = manifest["fps"].astype(str).str.strip()
        nonempty = fps_raw != ""
        fps = pd.to_numeric(fps_raw.where(nonempty), errors="coerce")
        bad = nonempty & (fps.isna() | (fps < 0))
        if bad.any():
            rows = manifest.index[bad].tolist()
            raise ValueError(f"Manifest: fps must be blank or numeric >= 0 at row(s) {rows}")
        manifest["fps"] = fps
    return manifest


def load_evaluation_summary(path: Path, expected_experiment: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation summary not found: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Evaluation summary must contain exactly one data row: {path}")
    missing = REQUIRED_SUMMARY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Evaluation summary {path.name} missing column(s): " + ", ".join(sorted(missing))
        )
    row = frame.iloc[0].copy()
    actual_experiment = str(row["experiment"]).strip()
    if actual_experiment != expected_experiment:
        raise ValueError(
            f"Experiment mismatch: manifest='{expected_experiment}' but "
            f"{path.name} contains '{actual_experiment}'"
        )

    for column in NUMERIC_SUMMARY_COLUMNS:
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            if column == "time_mae":
                row[column] = float("nan")
                continue
            raise ValueError(f"Evaluation summary {path.name}: '{column}' must be numeric")
        row[column] = float(value)

    for column in ("manual_total", "system_total", "absolute_error", "mean_class_absolute_error"):
        if float(row[column]) < 0:
            raise ValueError(f"Evaluation summary {path.name}: '{column}' must be >= 0")
    if not pd.isna(row["time_mae"]) and float(row["time_mae"]) < 0:
        raise ValueError(f"Evaluation summary {path.name}: 'time_mae' must be >= 0")
    return row


def build_results(manifest_path: Path) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, object]] = []
    metadata_columns = [c for c in manifest.columns if c != "evaluation_summary"]
    for _, manifest_row in manifest.iterrows():
        experiment = str(manifest_row["experiment"])
        summary_path = resolve_summary_path(str(manifest_row["evaluation_summary"]), manifest_path)
        summary = load_evaluation_summary(summary_path, experiment)

        item = {column: manifest_row[column] for column in metadata_columns}
        for column in NUMERIC_SUMMARY_COLUMNS:
            item[column] = summary[column]
        item["source_summary"] = portable_path(summary_path)
        rows.append(item)

    result = pd.DataFrame(rows)
    integer_columns = ["manual_total", "system_total", "signed_error", "absolute_error"]
    for column in integer_columns:
        result[column] = result[column].round().astype(int)
    return result


def save_results(results: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-experiment evaluation summaries into one master results CSV."
    )
    parser.add_argument("manifest_csv", type=Path, help="Experiment manifest CSV")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output CSV (default: evaluation/results_summary.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_results(args.manifest_csv)
    output = save_results(results, args.output)
    print(f"Experiments combined: {len(results)}")
    print(f"Saved: {output}")
    display_columns = [
        c for c in ("experiment", "detector", "tracker", "counting",
                    "counting_accuracy_pct", "time_mae", "fps")
        if c in results.columns
    ]
    if display_columns:
        print(results[display_columns].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
