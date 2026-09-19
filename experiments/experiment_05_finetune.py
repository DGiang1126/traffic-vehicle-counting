"""
E5 — Fine-tuning YOLOv8s for Vietnamese Traffic

Compare fairly on the SAME held-out test set:

1. Baseline:
   COCO-pretrained YOLOv8s
   COCO class IDs:
       bicycle    = 1
       car        = 2
       motorcycle = 3
       bus        = 5
       truck      = 7

2. Fine-tuned:
   YOLOv8s fine-tuned on the custom 5-class Vietnamese traffic dataset.

Important:
The custom dataset uses its own class IDs, so the baseline cannot be
evaluated directly against those label IDs. This experiment creates a
temporary COCO-ID version of the test labels before validating baseline.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PATHS
# ==========================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "vietnam_traffic"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"

BASELINE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8s.pt"
)

FINETUNED_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "vn_finetune"
    / "yolov8s_vn"
    / "weights"
    / "best.pt"
)


# ==========================================================
# E5 SETTINGS
# ==========================================================

IMAGE_SIZE = 640
BATCH_SIZE = 8

TARGET_CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "truck",
]


# ==========================================================
# OUTPUT
# ==========================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / f"e5_finetune_{RUN_ID}"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

BASELINE_VAL_DIR = OUTPUT_ROOT / "baseline_pretrained"
FINETUNED_VAL_DIR = OUTPUT_ROOT / "finetuned"

TEMP_DATASET_ROOT = (
    PROJECT_ROOT
    / "runtime"
    / "e5_baseline_coco"
)


# ==========================================================
# HELPERS
# ==========================================================

def load_custom_names() -> dict[int, str]:
    """
    Read class names from the custom Roboflow data.yaml.
    """

    with open(
        DATA_YAML,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    names = data["names"]

    if isinstance(names, list):
        return {
            index: name
            for index, name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(index): name
            for index, name in names.items()
        }

    raise ValueError(
        "Unsupported 'names' format in data.yaml"
    )


def normalize_model_names(
    names,
) -> dict[int, str]:

    if isinstance(names, dict):
        return {
            int(key): value
            for key, value in names.items()
        }

    return {
        index: name
        for index, name in enumerate(names)
    }


def invert_names(
    names: dict[int, str],
) -> dict[str, int]:

    return {
        name: class_id
        for class_id, name in names.items()
    }


def validate_paths():
    required = [
        DATA_YAML,
        TEST_IMAGES_DIR,
        TEST_LABELS_DIR,
        BASELINE_MODEL_PATH,
        FINETUNED_MODEL_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required path:\n{path}"
            )


def count_files(
    directory: Path,
    extensions: tuple[str, ...],
) -> int:

    return sum(
        1
        for file in directory.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in extensions
        )
    )


# ==========================================================
# PREPARE BASELINE TEST DATASET
# ==========================================================

def prepare_baseline_coco_dataset(
    custom_names: dict[int, str],
    coco_names: dict[int, str],
) -> tuple[Path, dict[str, int]]:

    """
    Convert custom test labels to corresponding COCO IDs.

    Example:

    custom:
        0 bicycle
        1 bus
        2 car
        3 motorcycle
        4 truck

    becomes:

    COCO:
        1 bicycle
        5 bus
        2 car
        3 motorcycle
        7 truck
    """

    coco_name_to_id = invert_names(
        coco_names
    )

    custom_to_coco: dict[int, int] = {}

    print("\n" + "=" * 70)
    print("CLASS MAPPING")
    print("=" * 70)

    for custom_id, class_name in (
        custom_names.items()
    ):

        if class_name not in TARGET_CLASSES:
            continue

        if class_name not in coco_name_to_id:
            raise ValueError(
                f"Class '{class_name}' "
                "not found in COCO model."
            )

        coco_id = coco_name_to_id[
            class_name
        ]

        custom_to_coco[
            custom_id
        ] = coco_id

        print(
            f"custom {custom_id:2d} "
            f"{class_name:12s} "
            f"-> COCO {coco_id:2d}"
        )

    mapped_names = {
        custom_names[custom_id]
        for custom_id in custom_to_coco
    }

    missing = (
        set(TARGET_CLASSES)
        - mapped_names
    )

    if missing:
        raise ValueError(
            "Missing custom classes: "
            + ", ".join(sorted(missing))
        )

    # Start clean
    if TEMP_DATASET_ROOT.exists():
        shutil.rmtree(
            TEMP_DATASET_ROOT
        )

    temp_images_dir = (
        TEMP_DATASET_ROOT
        / "test"
        / "images"
    )

    temp_labels_dir = (
        TEMP_DATASET_ROOT
        / "test"
        / "labels"
    )

    temp_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # Copy images
    # ----------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    copied_images = 0

    for image_path in (
        TEST_IMAGES_DIR.iterdir()
    ):

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in image_extensions
        ):
            shutil.copy2(
                image_path,
                temp_images_dir
                / image_path.name,
            )

            copied_images += 1

    # ----------------------------------------------
    # Remap labels
    # ----------------------------------------------

    converted_labels = 0
    converted_boxes = 0

    for label_path in (
        TEST_LABELS_DIR.glob("*.txt")
    ):

        output_path = (
            temp_labels_dir
            / label_path.name
        )

        output_lines = []

        text = label_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):

                parts = line.split()

                if len(parts) < 5:
                    raise ValueError(
                        f"Invalid YOLO label:\n"
                        f"{label_path}\n"
                        f"Line {line_number}: "
                        f"{line}"
                    )

                custom_id = int(
                    float(parts[0])
                )

                if custom_id not in (
                    custom_to_coco
                ):
                    raise ValueError(
                        f"Unknown custom class "
                        f"{custom_id} in:\n"
                        f"{label_path}"
                    )

                coco_id = (
                    custom_to_coco[
                        custom_id
                    ]
                )

                parts[0] = str(
                    coco_id
                )

                output_lines.append(
                    " ".join(parts)
                )

                converted_boxes += 1

        output_path.write_text(
            "\n".join(output_lines)
            + (
                "\n"
                if output_lines
                else ""
            ),
            encoding="utf-8",
        )

        converted_labels += 1

    print()
    print(
        f"Copied test images: "
        f"{copied_images}"
    )
    print(
        f"Converted labels: "
        f"{converted_labels}"
    )
    print(
        f"Converted boxes: "
        f"{converted_boxes}"
    )

    # ----------------------------------------------
    # COCO data.yaml
    # ----------------------------------------------

    temp_yaml_path = (
        TEMP_DATASET_ROOT
        / "data_coco.yaml"
    )

    yaml_data = {
        "path": str(
            TEMP_DATASET_ROOT.resolve()
        ),
        "train": "test/images",
        "val": "test/images",
        "test": "test/images",
        "names": coco_names,
    }

    with open(
        temp_yaml_path,
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            yaml_data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    target_coco_ids = {
        class_name: coco_name_to_id[
            class_name
        ]
        for class_name
        in TARGET_CLASSES
    }

    return (
        temp_yaml_path,
        target_coco_ids,
    )


# ==========================================================
# METRIC EXTRACTION
# ==========================================================

def get_class_ids_from_metrics(
    metrics,
    fallback_ids: list[int],
) -> list[int]:
    """
    Ultralytics versions differ slightly.
    Try to recover the class IDs represented by the metric arrays.
    """

    box = metrics.box

    if hasattr(
        box,
        "ap_class_index",
    ):
        values = (
            box.ap_class_index
        )

        if values is not None:
            return [
                int(value)
                for value in values
            ]

    # In our E5 test all five target
    # classes have ground-truth instances,
    # so this fallback order is valid.
    return list(
        fallback_ids
    )


def extract_metrics(
    metrics,
    class_names: dict[int, str],
    expected_ids: list[int],
) -> tuple[dict, dict]:

    box = metrics.box

    overall = {
        "precision": float(
            box.mp
        ),
        "recall": float(
            box.mr
        ),
        "map50": float(
            box.map50
        ),
        "map50_95": float(
            box.map
        ),
    }

    class_ids = (
        get_class_ids_from_metrics(
            metrics,
            expected_ids,
        )
    )

    precision_values = list(
        box.p
    )

    recall_values = list(
        box.r
    )

    map50_values = list(
        box.ap50
    )

    map_values = list(
        box.ap
    )

    lengths = {
        len(class_ids),
        len(precision_values),
        len(recall_values),
        len(map50_values),
        len(map_values),
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "Unable to align per-class "
            "Ultralytics metrics.\n"
            f"class_ids={len(class_ids)}, "
            f"P={len(precision_values)}, "
            f"R={len(recall_values)}, "
            f"AP50={len(map50_values)}, "
            f"AP={len(map_values)}"
        )

    per_class = {}

    for index, class_id in enumerate(
        class_ids
    ):

        class_name = class_names.get(
            class_id,
            f"class_{class_id}",
        )

        per_class[
            class_name
        ] = {
            "class_id": class_id,
            "precision": float(
                precision_values[
                    index
                ]
            ),
            "recall": float(
                recall_values[
                    index
                ]
            ),
            "map50": float(
                map50_values[
                    index
                ]
            ),
            "map50_95": float(
                map_values[
                    index
                ]
            ),
        }

    return (
        overall,
        per_class,
    )


# ==========================================================
# VALIDATION
# ==========================================================

def run_validation(
    model: YOLO,
    data_yaml: Path,
    class_ids: list[int],
    run_name: str,
):

    print("\n" + "=" * 70)
    print(
        f"VALIDATING: {run_name}"
    )
    print("=" * 70)

    return model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        classes=class_ids,

        # Keep both models under
        # identical validation settings.
        project=str(
            OUTPUT_ROOT
        ),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_overall_comparison(
    baseline: dict,
    finetuned: dict,
):

    path = (
        OUTPUT_ROOT
        / "e5_model_comparison.csv"
    )

    rows = [
        {
            "model": (
                "YOLOv8s_pretrained"
            ),
            **baseline,
        },
        {
            "model": (
                "YOLOv8s_finetuned"
            ),
            **finetuned,
        },
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "precision",
                "recall",
                "map50",
                "map50_95",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_per_class_comparison(
    baseline: dict,
    finetuned: dict,
):

    path = (
        OUTPUT_ROOT
        / "e5_per_class_comparison.csv"
    )

    rows = []

    for class_name in TARGET_CLASSES:

        base = baseline.get(
            class_name,
            {},
        )

        fine = finetuned.get(
            class_name,
            {},
        )

        row = {
            "class": class_name,

            "baseline_precision": (
                base.get(
                    "precision",
                    0.0,
                )
            ),
            "finetuned_precision": (
                fine.get(
                    "precision",
                    0.0,
                )
            ),

            "baseline_recall": (
                base.get(
                    "recall",
                    0.0,
                )
            ),
            "finetuned_recall": (
                fine.get(
                    "recall",
                    0.0,
                )
            ),

            "baseline_map50": (
                base.get(
                    "map50",
                    0.0,
                )
            ),
            "finetuned_map50": (
                fine.get(
                    "map50",
                    0.0,
                )
            ),

            "delta_map50": (
                fine.get(
                    "map50",
                    0.0,
                )
                -
                base.get(
                    "map50",
                    0.0,
                )
            ),

            "baseline_map50_95": (
                base.get(
                    "map50_95",
                    0.0,
                )
            ),
            "finetuned_map50_95": (
                fine.get(
                    "map50_95",
                    0.0,
                )
            ),

            "delta_map50_95": (
                fine.get(
                    "map50_95",
                    0.0,
                )
                -
                base.get(
                    "map50_95",
                    0.0,
                )
            ),
        }

        rows.append(
            row
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_json_summary(
    baseline_overall,
    finetuned_overall,
    baseline_per_class,
    finetuned_per_class,
    custom_names,
    coco_target_ids,
):

    path = (
        OUTPUT_ROOT
        / "e5_summary.json"
    )

    data = {
        "experiment": (
            "E5 - Fine-tuning YOLOv8s "
            "for Vietnamese Traffic"
        ),
        "test_images": 20,
        "image_size": IMAGE_SIZE,
        "baseline_model": str(
            BASELINE_MODEL_PATH
        ),
        "finetuned_model": str(
            FINETUNED_MODEL_PATH
        ),
        "custom_names": (
            custom_names
        ),
        "baseline_coco_ids": (
            coco_target_ids
        ),
        "baseline": {
            "overall": (
                baseline_overall
            ),
            "per_class": (
                baseline_per_class
            ),
        },
        "finetuned": {
            "overall": (
                finetuned_overall
            ),
            "per_class": (
                finetuned_per_class
            ),
        },
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


# ==========================================================
# PRINT
# ==========================================================

def percent(
    value: float,
) -> str:

    return (
        f"{value * 100:.2f}%"
    )


def print_overall(
    baseline: dict,
    finetuned: dict,
):

    print("\n")
    print("=" * 86)
    print(
        "E5 FINAL MODEL COMPARISON"
    )
    print("=" * 86)

    print(
        f"{'Model':24s}"
        f"{'Precision':>14s}"
        f"{'Recall':>14s}"
        f"{'mAP50':>14s}"
        f"{'mAP50-95':>16s}"
    )

    print("-" * 86)

    for name, values in [
        (
            "YOLOv8s pretrained",
            baseline,
        ),
        (
            "YOLOv8s fine-tuned",
            finetuned,
        ),
    ]:

        print(
            f"{name:24s}"
            f"{percent(values['precision']):>14s}"
            f"{percent(values['recall']):>14s}"
            f"{percent(values['map50']):>14s}"
            f"{percent(values['map50_95']):>16s}"
        )


def print_per_class(
    baseline: dict,
    finetuned: dict,
):

    print("\n")
    print("=" * 86)
    print(
        "E5 PER-CLASS mAP50"
    )
    print("=" * 86)

    print(
        f"{'Class':15s}"
        f"{'Baseline':>16s}"
        f"{'Fine-tuned':>16s}"
        f"{'Delta':>16s}"
    )

    print("-" * 63)

    for class_name in (
        TARGET_CLASSES
    ):

        base = baseline.get(
            class_name,
            {},
        ).get(
            "map50",
            0.0,
        )

        fine = finetuned.get(
            class_name,
            {},
        ).get(
            "map50",
            0.0,
        )

        delta = (
            fine - base
        )

        print(
            f"{class_name:15s}"
            f"{percent(base):>16s}"
            f"{percent(fine):>16s}"
            f"{delta * 100:>+15.2f}pp"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    validate_paths()

    image_count = count_files(
        TEST_IMAGES_DIR,
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ),
    )

    label_count = count_files(
        TEST_LABELS_DIR,
        (".txt",),
    )

    print("\n" + "=" * 70)
    print("E5 — DATASET CHECK")
    print("=" * 70)

    print(
        f"Test images: {image_count}"
    )
    print(
        f"Test labels: {label_count}"
    )

    if image_count != 20:
        print(
            "WARNING: Expected 20 "
            "test images."
        )

    if image_count != label_count:
        raise ValueError(
            "Number of test images "
            "and labels does not match."
        )

    # ======================================================
    # LOAD MODELS
    # ======================================================

    print("\nLoading baseline YOLOv8s...")

    baseline_model = YOLO(
        str(BASELINE_MODEL_PATH)
    )

    print(
        "Loading fine-tuned YOLOv8s..."
    )

    finetuned_model = YOLO(
        str(FINETUNED_MODEL_PATH)
    )

    # ======================================================
    # CLASS NAMES
    # ======================================================

    custom_names = (
        load_custom_names()
    )

    coco_names = (
        normalize_model_names(
            baseline_model.names
        )
    )

    finetuned_names = (
        normalize_model_names(
            finetuned_model.names
        )
    )

    print("\nCustom classes:")

    for class_id, class_name in (
        custom_names.items()
    ):
        print(
            f"  {class_id}: "
            f"{class_name}"
        )

    print("\nFine-tuned model classes:")

    for class_id, class_name in (
        finetuned_names.items()
    ):
        print(
            f"  {class_id}: "
            f"{class_name}"
        )

    # ======================================================
    # PREP BASELINE LABELS
    # ======================================================

    (
        coco_test_yaml,
        coco_target_ids,
    ) = prepare_baseline_coco_dataset(
        custom_names,
        coco_names,
    )

    baseline_class_ids = sorted(
        coco_target_ids.values()
    )

    custom_name_to_id = (
        invert_names(
            custom_names
        )
    )

    finetuned_class_ids = sorted(
        custom_name_to_id[
            class_name
        ]
        for class_name
        in TARGET_CLASSES
    )

    # ======================================================
    # BASELINE VALIDATION
    # ======================================================

    baseline_metrics = (
        run_validation(
            model=baseline_model,
            data_yaml=coco_test_yaml,
            class_ids=baseline_class_ids,
            run_name=(
                "baseline_pretrained"
            ),
        )
    )

    (
        baseline_overall,
        baseline_per_class,
    ) = extract_metrics(
        baseline_metrics,
        coco_names,
        baseline_class_ids,
    )

    # ======================================================
    # FINE-TUNED VALIDATION
    # ======================================================

    finetuned_metrics = (
        run_validation(
            model=finetuned_model,
            data_yaml=DATA_YAML,
            class_ids=(
                finetuned_class_ids
            ),
            run_name=(
                "finetuned"
            ),
        )
    )

    (
        finetuned_overall,
        finetuned_per_class,
    ) = extract_metrics(
        finetuned_metrics,
        finetuned_names,
        finetuned_class_ids,
    )

    # ======================================================
    # SAVE
    # ======================================================

    comparison_csv = (
        save_overall_comparison(
            baseline_overall,
            finetuned_overall,
        )
    )

    per_class_csv = (
        save_per_class_comparison(
            baseline_per_class,
            finetuned_per_class,
        )
    )

    summary_json = (
        save_json_summary(
            baseline_overall,
            finetuned_overall,
            baseline_per_class,
            finetuned_per_class,
            custom_names,
            coco_target_ids,
        )
    )

    # ======================================================
    # PRINT
    # ======================================================

    print_overall(
        baseline_overall,
        finetuned_overall,
    )

    print_per_class(
        baseline_per_class,
        finetuned_per_class,
    )

    print("\n")
    print("=" * 86)
    print("OUTPUT")
    print("=" * 86)

    print(
        "Run directory:"
    )
    print(
        OUTPUT_ROOT
    )

    print(
        "\nOverall comparison:"
    )
    print(
        comparison_csv
    )

    print(
        "\nPer-class comparison:"
    )
    print(
        per_class_csv
    )

    print(
        "\nJSON summary:"
    )
    print(
        summary_json
    )

    print(
        "\nTemporary COCO-remapped "
        "dataset:"
    )
    print(
        TEMP_DATASET_ROOT
    )


if __name__ == "__main__":
    main()