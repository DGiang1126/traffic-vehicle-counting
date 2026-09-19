"""
E5 — In-domain Diagnostic Evaluation

Purpose
-------
Compare:

1. YOLOv8s pretrained on COCO
2. YOLOv8s fine-tuned on Vietnamese traffic

Evaluation data:
    cam01_t02 = VALIDATION SET = 20 images

This experiment is diagnostic only because this validation set
was already used during fine-tuning / best.pt selection.

It answers:
    "Did fine-tuning improve performance inside the same cam01 domain?"

It does NOT replace the held-out cam02 test result.
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
# DATASET PATHS
# ==========================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "vietnam_traffic"
)

DATA_YAML = (
    DATASET_ROOT
    / "data.yaml"
)

# ----------------------------------------------------------
# IMPORTANT:
# E5 IN-DOMAIN uses VALID = cam01_t02
# ----------------------------------------------------------

VAL_IMAGES_DIR = (
    DATASET_ROOT
    / "valid"
    / "images"
)

VAL_LABELS_DIR = (
    DATASET_ROOT
    / "valid"
    / "labels"
)


# ==========================================================
# MODEL PATHS
# ==========================================================

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
# SETTINGS
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

RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / f"e5_indomain_{RUN_ID}"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# Completely separate temporary dataset
TEMP_DATASET_ROOT = (
    PROJECT_ROOT
    / "runtime"
    / "e5_indomain_baseline_coco"
)


# ==========================================================
# HELPER — LOAD CUSTOM CLASS NAMES
# ==========================================================

def load_custom_names() -> dict[int, str]:

    with open(
        DATA_YAML,
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(
            file
        )

    names = data["names"]

    if isinstance(
        names,
        list,
    ):

        return {
            index: name
            for index, name
            in enumerate(names)
        }

    if isinstance(
        names,
        dict,
    ):

        return {
            int(index): name
            for index, name
            in names.items()
        }

    raise ValueError(
        "Unsupported 'names' "
        "format in data.yaml"
    )


# ==========================================================
# HELPER — NORMALIZE MODEL NAMES
# ==========================================================

def normalize_model_names(
    names,
) -> dict[int, str]:

    if isinstance(
        names,
        dict,
    ):

        return {
            int(class_id): class_name
            for class_id, class_name
            in names.items()
        }

    return {
        index: class_name
        for index, class_name
        in enumerate(names)
    }


def invert_names(
    names: dict[int, str],
) -> dict[str, int]:

    return {
        class_name: class_id
        for class_id, class_name
        in names.items()
    }


# ==========================================================
# VALIDATE PATHS
# ==========================================================

def validate_paths():

    required_paths = [
        DATA_YAML,
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        BASELINE_MODEL_PATH,
        FINETUNED_MODEL_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required path:\n"
                f"{path}"
            )


# ==========================================================
# COUNT FILES
# ==========================================================

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
# CREATE COCO-ID VERSION OF VALIDATION LABELS
# ==========================================================

def prepare_baseline_coco_dataset(
    custom_names: dict[int, str],
    coco_names: dict[int, str],
) -> tuple[Path, dict[str, int]]:

    """
    Convert custom validation labels:

        custom 0 bicycle
        custom 1 bus
        custom 2 car
        custom 3 motorcycle
        custom 4 truck

    into COCO IDs:

        COCO 1 bicycle
        COCO 5 bus
        COCO 2 car
        COCO 3 motorcycle
        COCO 7 truck

    Original dataset is NOT modified.
    """

    coco_name_to_id = (
        invert_names(
            coco_names
        )
    )

    custom_to_coco: dict[
        int,
        int,
    ] = {}

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLASS MAPPING"
    )

    print(
        "=" * 70
    )

    for (
        custom_id,
        class_name,
    ) in custom_names.items():

        if (
            class_name
            not in TARGET_CLASSES
        ):
            continue

        if (
            class_name
            not in coco_name_to_id
        ):

            raise ValueError(
                f"Class '{class_name}' "
                "not found in COCO."
            )

        coco_id = (
            coco_name_to_id[
                class_name
            ]
        )

        custom_to_coco[
            custom_id
        ] = coco_id

        print(
            f"custom "
            f"{custom_id:2d} "
            f"{class_name:12s} "
            f"-> COCO "
            f"{coco_id:2d}"
        )

    mapped_names = {
        custom_names[
            custom_id
        ]
        for custom_id
        in custom_to_coco
    }

    missing_classes = (
        set(TARGET_CLASSES)
        - mapped_names
    )

    if missing_classes:

        raise ValueError(
            "Missing custom classes: "
            + ", ".join(
                sorted(
                    missing_classes
                )
            )
        )

    # ------------------------------------------------------
    # CLEAN PREVIOUS TEMP DATA
    # ------------------------------------------------------

    if TEMP_DATASET_ROOT.exists():

        shutil.rmtree(
            TEMP_DATASET_ROOT
        )

    temp_images_dir = (
        TEMP_DATASET_ROOT
        / "valid"
        / "images"
    )

    temp_labels_dir = (
        TEMP_DATASET_ROOT
        / "valid"
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

    # ------------------------------------------------------
    # COPY VALIDATION IMAGES
    # ------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    copied_images = 0

    for image_path in (
        VAL_IMAGES_DIR.iterdir()
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

    # ------------------------------------------------------
    # REMAP LABEL IDs
    # ------------------------------------------------------

    converted_labels = 0
    converted_boxes = 0

    for label_path in (
        VAL_LABELS_DIR.glob(
            "*.txt"
        )
    ):

        output_path = (
            temp_labels_dir
            / label_path.name
        )

        output_lines = []

        text = (
            label_path
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if text:

            for (
                line_number,
                line,
            ) in enumerate(
                text.splitlines(),
                start=1,
            ):

                parts = (
                    line.split()
                )

                if len(parts) < 5:

                    raise ValueError(
                        "Invalid YOLO "
                        "label:\n"
                        f"{label_path}\n"
                        f"Line "
                        f"{line_number}: "
                        f"{line}"
                    )

                custom_id = int(
                    float(
                        parts[0]
                    )
                )

                if (
                    custom_id
                    not in custom_to_coco
                ):

                    raise ValueError(
                        f"Unknown custom "
                        f"class ID "
                        f"{custom_id}\n"
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
                    " ".join(
                        parts
                    )
                )

                converted_boxes += 1

        output_path.write_text(
            (
                "\n".join(
                    output_lines
                )
                +
                (
                    "\n"
                    if output_lines
                    else ""
                )
            ),
            encoding="utf-8",
        )

        converted_labels += 1

    print()

    print(
        "Copied validation "
        f"images: "
        f"{copied_images}"
    )

    print(
        "Converted labels: "
        f"{converted_labels}"
    )

    print(
        "Converted boxes: "
        f"{converted_boxes}"
    )

    # ------------------------------------------------------
    # CREATE TEMP COCO YAML
    # ------------------------------------------------------

    temp_yaml_path = (
        TEMP_DATASET_ROOT
        / "data_coco_indomain.yaml"
    )

    yaml_data = {
        "path": str(
            TEMP_DATASET_ROOT.resolve()
        ),

        "train": (
            "valid/images"
        ),

        "val": (
            "valid/images"
        ),

        "test": (
            "valid/images"
        ),

        "names": (
            coco_names
        ),
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
        class_name: (
            coco_name_to_id[
                class_name
            ]
        )
        for class_name
        in TARGET_CLASSES
    }

    return (
        temp_yaml_path,
        target_coco_ids,
    )


# ==========================================================
# GET ULTRALYTICS CLASS IDS
# ==========================================================

def get_metric_class_ids(
    metrics,
    fallback_ids: list[int],
) -> list[int]:

    box = metrics.box

    if hasattr(
        box,
        "ap_class_index",
    ):

        class_indices = (
            box.ap_class_index
        )

        if (
            class_indices
            is not None
        ):

            return [
                int(value)
                for value
                in class_indices
            ]

    return list(
        fallback_ids
    )


# ==========================================================
# EXTRACT METRICS
# ==========================================================

def extract_metrics(
    metrics,
    model_names: dict[int, str],
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
        get_metric_class_ids(
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
            "Unable to align "
            "Ultralytics "
            "per-class metrics.\n"
            f"class_ids="
            f"{len(class_ids)}, "
            f"P="
            f"{len(precision_values)}, "
            f"R="
            f"{len(recall_values)}, "
            f"AP50="
            f"{len(map50_values)}, "
            f"AP="
            f"{len(map_values)}"
        )

    per_class = {}

    for (
        index,
        class_id,
    ) in enumerate(
        class_ids
    ):

        class_name = (
            model_names.get(
                class_id,
                f"class_{class_id}",
            )
        )

        per_class[
            class_name
        ] = {
            "class_id": (
                class_id
            ),

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
# RUN VALIDATION
# ==========================================================

def run_validation(
    model: YOLO,
    data_yaml: Path,
    class_ids: list[int],
    run_name: str,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"VALIDATING: "
        f"{run_name}"
    )

    print(
        "=" * 70
    )

    return model.val(
        data=str(
            data_yaml
        ),

        # IMPORTANT:
        # cam01_t02 = VALID
        split="val",

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        classes=class_ids,

        project=str(
            OUTPUT_ROOT
        ),

        name=run_name,

        exist_ok=True,

        plots=True,

        verbose=True,
    )


# ==========================================================
# SAVE OVERALL COMPARISON
# ==========================================================

def save_overall_comparison(
    baseline: dict,
    finetuned: dict,
):

    path = (
        OUTPUT_ROOT
        / "e5_indomain_model_comparison.csv"
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

        writer.writerows(
            rows
        )

    return path


# ==========================================================
# SAVE PER-CLASS COMPARISON
# ==========================================================

def save_per_class_comparison(
    baseline: dict,
    finetuned: dict,
):

    path = (
        OUTPUT_ROOT
        / "e5_indomain_per_class_comparison.csv"
    )

    rows = []

    for class_name in (
        TARGET_CLASSES
    ):

        baseline_values = (
            baseline.get(
                class_name,
                {},
            )
        )

        finetuned_values = (
            finetuned.get(
                class_name,
                {},
            )
        )

        baseline_map50 = (
            baseline_values.get(
                "map50",
                0.0,
            )
        )

        finetuned_map50 = (
            finetuned_values.get(
                "map50",
                0.0,
            )
        )

        baseline_map5095 = (
            baseline_values.get(
                "map50_95",
                0.0,
            )
        )

        finetuned_map5095 = (
            finetuned_values.get(
                "map50_95",
                0.0,
            )
        )

        rows.append(
            {
                "class": (
                    class_name
                ),

                "baseline_precision": (
                    baseline_values.get(
                        "precision",
                        0.0,
                    )
                ),

                "finetuned_precision": (
                    finetuned_values.get(
                        "precision",
                        0.0,
                    )
                ),

                "baseline_recall": (
                    baseline_values.get(
                        "recall",
                        0.0,
                    )
                ),

                "finetuned_recall": (
                    finetuned_values.get(
                        "recall",
                        0.0,
                    )
                ),

                "baseline_map50": (
                    baseline_map50
                ),

                "finetuned_map50": (
                    finetuned_map50
                ),

                "delta_map50": (
                    finetuned_map50
                    - baseline_map50
                ),

                "baseline_map50_95": (
                    baseline_map5095
                ),

                "finetuned_map50_95": (
                    finetuned_map5095
                ),

                "delta_map50_95": (
                    finetuned_map5095
                    - baseline_map5095
                ),
            }
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return path


# ==========================================================
# SAVE JSON SUMMARY
# ==========================================================

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
        / "e5_indomain_summary.json"
    )

    summary = {
        "experiment": (
            "E5 In-domain "
            "Diagnostic"
        ),

        "evaluation_domain": (
            "cam01_t02"
        ),

        "evaluation_split": (
            "validation"
        ),

        "note": (
            "Diagnostic only. "
            "This validation set "
            "was used during "
            "fine-tuning."
        ),

        "image_size": (
            IMAGE_SIZE
        ),

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
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


# ==========================================================
# PRINT HELPERS
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

    print(
        "=" * 86
    )

    print(
        "E5 IN-DOMAIN "
        "MODEL COMPARISON"
    )

    print(
        "=" * 86
    )

    print(
        f"{'Model':24s}"
        f"{'Precision':>14s}"
        f"{'Recall':>14s}"
        f"{'mAP50':>14s}"
        f"{'mAP50-95':>16s}"
    )

    print(
        "-" * 86
    )

    for (
        name,
        values,
    ) in [

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

    print(
        "=" * 70
    )

    print(
        "E5 IN-DOMAIN "
        "PER-CLASS mAP50"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Class':15s}"
        f"{'Baseline':>16s}"
        f"{'Fine-tuned':>16s}"
        f"{'Delta':>16s}"
    )

    print(
        "-" * 63
    )

    for class_name in (
        TARGET_CLASSES
    ):

        baseline_map50 = (
            baseline.get(
                class_name,
                {},
            ).get(
                "map50",
                0.0,
            )
        )

        finetuned_map50 = (
            finetuned.get(
                class_name,
                {},
            ).get(
                "map50",
                0.0,
            )
        )

        delta = (
            finetuned_map50
            - baseline_map50
        )

        print(
            f"{class_name:15s}"
            f"{percent(baseline_map50):>16s}"
            f"{percent(finetuned_map50):>16s}"
            f"{delta * 100:>+15.2f}pp"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # CHECK PATHS
    # ------------------------------------------------------

    validate_paths()

    # ------------------------------------------------------
    # CHECK DATASET SIZE
    # ------------------------------------------------------

    image_count = (
        count_files(
            VAL_IMAGES_DIR,
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            ),
        )
    )

    label_count = (
        count_files(
            VAL_LABELS_DIR,
            (
                ".txt",
            ),
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "E5 IN-DOMAIN "
        "DATASET CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"Validation images: "
        f"{image_count}"
    )

    print(
        f"Validation labels: "
        f"{label_count}"
    )

    if (
        image_count
        != label_count
    ):

        raise ValueError(
            "Number of validation "
            "images and labels "
            "does not match."
        )

    if image_count != 20:

        print(
            "WARNING: Expected "
            "20 validation images."
        )

    # ------------------------------------------------------
    # LOAD MODELS
    # ------------------------------------------------------

    print(
        "\nLoading baseline "
        "YOLOv8s..."
    )

    baseline_model = YOLO(
        str(
            BASELINE_MODEL_PATH
        )
    )

    print(
        "Loading fine-tuned "
        "YOLOv8s..."
    )

    finetuned_model = YOLO(
        str(
            FINETUNED_MODEL_PATH
        )
    )

    # ------------------------------------------------------
    # CLASS NAMES
    # ------------------------------------------------------

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

    print(
        "\nCustom classes:"
    )

    for (
        class_id,
        class_name,
    ) in custom_names.items():

        print(
            f"  {class_id}: "
            f"{class_name}"
        )

    print(
        "\nFine-tuned "
        "model classes:"
    )

    for (
        class_id,
        class_name,
    ) in finetuned_names.items():

        print(
            f"  {class_id}: "
            f"{class_name}"
        )

    # ------------------------------------------------------
    # PREPARE COCO LABEL VERSION
    # ------------------------------------------------------

    (
        coco_val_yaml,
        coco_target_ids,
    ) = (
        prepare_baseline_coco_dataset(
            custom_names,
            coco_names,
        )
    )

    baseline_class_ids = (
        sorted(
            coco_target_ids.values()
        )
    )

    custom_name_to_id = (
        invert_names(
            custom_names
        )
    )

    finetuned_class_ids = (
        sorted(
            custom_name_to_id[
                class_name
            ]
            for class_name
            in TARGET_CLASSES
        )
    )

    # ------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------

    baseline_metrics = (
        run_validation(
            model=baseline_model,

            data_yaml=(
                coco_val_yaml
            ),

            class_ids=(
                baseline_class_ids
            ),

            run_name=(
                "baseline_pretrained_indomain"
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

    # ------------------------------------------------------
    # FINE-TUNED
    # ------------------------------------------------------

    finetuned_metrics = (
        run_validation(
            model=finetuned_model,

            data_yaml=(
                DATA_YAML
            ),

            class_ids=(
                finetuned_class_ids
            ),

            run_name=(
                "finetuned_indomain"
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

    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

    overall_csv = (
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

    # ------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------

    print_overall(
        baseline_overall,
        finetuned_overall,
    )

    print_per_class(
        baseline_per_class,
        finetuned_per_class,
    )

    print("\n")

    print(
        "=" * 86
    )

    print(
        "OUTPUT"
    )

    print(
        "=" * 86
    )

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
        overall_csv
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
        "\nTemporary "
        "COCO-remapped "
        "validation dataset:"
    )

    print(
        TEMP_DATASET_ROOT
    )


if __name__ == "__main__":
    main()