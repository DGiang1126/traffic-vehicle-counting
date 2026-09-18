"""
E4 — Counting Strategy Comparison

Compare:
    1) Single-line counting
    2) Multi-line counting

Same video, detector, tracker and tracked objects.
Only counting strategy changes.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import torch


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

from src.tracking.tracker import VehicleTracker
from src.counting import LineDefinition, MultiLineCounter
from src.statistics.statistics import run_statistics

from evaluation.evaluate_counting import (
    run_evaluation,
    counting_accuracy,
)


# ==========================================================
# E4 INPUT
# ==========================================================

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "evaluation"
    / "eval_02.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8s.pt"
)


# ==========================================================
# COUNTING CONFIGS
# ==========================================================

# Single-line:
# 1 line phủ toàn bộ luồng cần đếm.
SINGLE_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "line_configs"
    / "eval_02_e4_single_line.json"
)

# Multi-line:
# 2 line rời nhau:
# - Cong_chinh
# - Nhanh_phu
#
# PHẢI là đúng 2 line đã dùng khi manual GT = 180 + 19.
MULTI_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "line_configs"
    / "eval_02_e4_multi_line.json"
)


# ==========================================================
# FIXED EXPERIMENT SETTINGS
# ==========================================================

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640

TRACKER_TYPE = "bytetrack"

# Giữ đúng counting rule hiện tại của integration.
ALLOWED_DIRECTION = "negative_to_positive"


# ==========================================================
# GROUND TRUTH
# ==========================================================
#
# SINGLE-LINE GT
#
# Ground Truth Kiệt:
# TOTAL = 199
#
# ==========================================================

SINGLE_GT = {
    "car": 178,
    "motorcycle": 2,
    "bus": 2,
    "truck": 17,
    "bicycle": 0,
}


# ==========================================================
# MULTI-LINE GT
# ==========================================================
#
# Bạn vừa manual:
#
# Cong_chinh:
#   car        161
#   motorcycle 2
#   bus        2
#   truck      15
#   bicycle    0
#   total      180
#
# Nhanh_phu:
#   car        17
#   motorcycle 0
#   bus        0
#   truck       2
#   bicycle     0
#   total       19
#
# ==========================================================

MULTI_GT_BY_LINE = {

    "Cong_chinh": {
        "car": 161,
        "motorcycle": 2,
        "bus": 2,
        "truck": 15,
        "bicycle": 0,
    },

    "Nhanh_phu": {
        "car": 17,
        "motorcycle": 0,
        "bus": 0,
        "truck": 2,
        "bicycle": 0,
    },
}


VEHICLE_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
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
    / f"e4_eval02_{RUN_ID}"
)

SINGLE_OUTPUT_DIR = (
    OUTPUT_ROOT
    / "single_line"
)

MULTI_OUTPUT_DIR = (
    OUTPUT_ROOT
    / "multi_line"
)

SINGLE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MULTI_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SINGLE_EVENTS_PATH = (
    SINGLE_OUTPUT_DIR
    / "e4_eval02_single_line_events.csv"
)

MULTI_EVENTS_PATH = (
    MULTI_OUTPUT_DIR
    / "e4_eval02_multi_line_events.csv"
)

OUTPUT_VIDEO_PATH = (
    OUTPUT_ROOT
    / "e4_eval02_comparison.mp4"
)


# ==========================================================
# PREVIEW
# ==========================================================

SHOW_PREVIEW = True
SAVE_OUTPUT_VIDEO = True

PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720

WINDOW_NAME = (
    "E4 eval_02 | Single-line vs Multi-line | Q = quit"
)


# ==========================================================
# DEVICE
# ==========================================================

if torch.cuda.is_available():

    DEVICE = "cuda:0"

else:

    DEVICE = "cpu"


# ==========================================================
# HELPERS
# ==========================================================

def total_from_counts(
    counts: dict[str, int],
) -> int:

    return sum(
        int(
            counts.get(
                cls,
                0,
            )
        )
        for cls in VEHICLE_CLASSES
    )


def build_multi_gt_total() -> dict[str, int]:

    result = {
        cls: 0
        for cls in VEHICLE_CLASSES
    }

    for line_counts in (
        MULTI_GT_BY_LINE.values()
    ):

        for cls in VEHICLE_CLASSES:

            result[cls] += int(
                line_counts.get(
                    cls,
                    0,
                )
            )

    return result


MULTI_GT = build_multi_gt_total()


# ==========================================================
# LOAD LINE CONFIG
# ==========================================================

def load_lines(
    config_path: Path,
) -> list[LineDefinition]:

    if not config_path.exists():

        raise FileNotFoundError(
            f"Khong tim thay config:\n{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    if "lines" not in config:

        raise ValueError(
            f"Config khong co key 'lines': "
            f"{config_path}"
        )

    lines: list[LineDefinition] = []

    for item in config["lines"]:

        start = tuple(
            map(
                int,
                item["start"],
            )
        )

        end = tuple(
            map(
                int,
                item["end"],
            )
        )

        line = LineDefinition(
            name=item["name"],
            start=start,
            end=end,

            negative_to_positive=(
                "negative_to_positive"
            ),

            positive_to_negative=(
                "positive_to_negative"
            ),
        )

        lines.append(
            line
        )

    return lines


# ==========================================================
# RESIZE PREVIEW
# ==========================================================

def resize_for_preview(
    frame,
):

    height, width = frame.shape[:2]

    scale = min(
        PREVIEW_MAX_WIDTH / width,
        PREVIEW_MAX_HEIGHT / height,
        1.0,
    )

    if scale >= 1.0:

        return frame

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ==========================================================
# EVENT CSV
# ==========================================================

EVENT_FIELDS = [
    "timestamp",
    "frame_index",
    "track_id",
    "class",
    "line",
    "direction",
    "confidence",
]


def open_event_writer(
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file = open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        file,
        fieldnames=EVENT_FIELDS,
    )

    writer.writeheader()

    return file, writer


# ==========================================================
# GT SNAPSHOT
# ==========================================================

def save_gt_snapshot(
    path: Path,
    counts: dict[str, int],
):

    row = {
        "minute": 0,
        "motorcycle": counts.get(
            "motorcycle",
            0,
        ),
        "car": counts.get(
            "car",
            0,
        ),
        "bus": counts.get(
            "bus",
            0,
        ),
        "truck": counts.get(
            "truck",
            0,
        ),
        "bicycle": counts.get(
            "bicycle",
            0,
        ),
        "total": total_from_counts(
            counts
        ),
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def save_gt_by_line_snapshot():

    rows = []

    # Single-line
    single_row = {
        "strategy": "single_line",
        "line": "single_line",
    }

    for cls in VEHICLE_CLASSES:

        single_row[
            cls
        ] = SINGLE_GT[
            cls
        ]

    single_row[
        "total"
    ] = total_from_counts(
        SINGLE_GT
    )

    rows.append(
        single_row
    )

    # Multi-line
    for (
        line_name,
        counts,
    ) in MULTI_GT_BY_LINE.items():

        row = {
            "strategy": "multi_line",
            "line": line_name,
        }

        for cls in VEHICLE_CLASSES:

            row[
                cls
            ] = counts[
                cls
            ]

        row[
            "total"
        ] = total_from_counts(
            counts
        )

        rows.append(
            row
        )

    path = (
        OUTPUT_ROOT
        / "e4_ground_truth_by_line.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


# ==========================================================
# COUNT EVENT HELPER
# ==========================================================

def save_events(
    events,
    writer,
    class_counts,
    line_counts,
):

    total_new = 0

    for event in events:

        writer.writerow(
            event.to_dict()
        )

        class_counts[
            event.class_name
        ] += 1

        line_counts[
            event.line
        ] += 1

        total_new += 1

    return total_new


# ==========================================================
# DRAW LINE
# ==========================================================

def draw_line(
    frame,
    line: LineDefinition,
    color,
    label_prefix: str,
):

    cv2.line(
        frame,
        line.start,
        line.end,
        color,
        4,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        line.start,
        7,
        color,
        -1,
    )

    cv2.circle(
        frame,
        line.end,
        7,
        color,
        -1,
    )

    label_position = (
        line.start[0] + 10,
        max(
            line.start[1] - 10,
            25,
        ),
    )

    cv2.putText(
        frame,
        (
            f"{label_prefix}"
            f"{line.name}"
        ),
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


# ==========================================================
# RUN VIDEO ONCE
# ==========================================================

def run_e4_pipeline(
    single_lines: list[LineDefinition],
    multi_lines: list[LineDefinition],
):

    print("\n" + "=" * 70)
    print("E4 — SINGLE-LINE VS MULTI-LINE")
    print("=" * 70)

    print(
        "Video:",
        VIDEO_PATH,
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Tracker:",
        TRACKER_TYPE,
    )

    print(
        "conf:",
        CONF_THRESHOLD,
    )

    print(
        "imgsz:",
        IMAGE_SIZE,
    )

    print(
        "Device:",
        DEVICE,
    )

    print("\nSingle-line config:")

    for line in single_lines:

        print(
            f"  {line.name}: "
            f"{line.start} -> {line.end}"
        )

    print("\nMulti-line config:")

    for line in multi_lines:

        print(
            f"  {line.name}: "
            f"{line.start} -> {line.end}"
        )

    # ======================================================
    # TWO INDEPENDENT COUNTERS — DUC ANH
    # ======================================================

    single_counter = MultiLineCounter(
        single_lines,
        allowed_direction=(
            ALLOWED_DIRECTION
        ),
    )

    multi_counter = MultiLineCounter(
        multi_lines,
        allowed_direction=(
            ALLOWED_DIRECTION
        ),
    )

    # ======================================================
    # TRACKER — THUAN
    # ======================================================

    tracker = VehicleTracker(
        model_path=MODEL_PATH,
        tracker_type=TRACKER_TYPE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
    )

    tracker.model.to(
        DEVICE
    )

    # ======================================================
    # VIDEO
    # ======================================================

    cap = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Khong mo duoc video: "
            f"{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        "\nResolution:",
        f"{width}x{height}",
    )

    print(
        "FPS:",
        fps,
    )

    print(
        "Frames:",
        total_frames,
    )

    # ======================================================
    # CSV
    # ======================================================

    (
        single_file,
        single_writer,
    ) = open_event_writer(
        SINGLE_EVENTS_PATH
    )

    (
        multi_file,
        multi_writer,
    ) = open_event_writer(
        MULTI_EVENTS_PATH
    )

    # ======================================================
    # COUNTS
    # ======================================================

    single_class_counts = defaultdict(
        int
    )

    multi_class_counts = defaultdict(
        int
    )

    single_line_counts = defaultdict(
        int
    )

    multi_line_counts = defaultdict(
        int
    )

    single_total = 0
    multi_total = 0

    frame_index = 0
    aborted = False

    # ======================================================
    # VIDEO WRITER
    # ======================================================

    video_writer = None

    if SAVE_OUTPUT_VIDEO:

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        video_writer = cv2.VideoWriter(
            str(
                OUTPUT_VIDEO_PATH
            ),
            fourcc,
            fps,
            (
                width,
                height,
            ),
        )

    # ======================================================
    # PREVIEW
    # ======================================================

    if SHOW_PREVIEW:

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

    # ======================================================
    # PROCESS
    # ======================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            # ==============================================
            # TRACKING
            # ==============================================

            objects = tracker.track_frame(
                frame,
                frame_number=frame_index,
                fps=fps,
            )

            # ==============================================
            # SAME TRACKS -> BOTH COUNTING STRATEGIES
            # ==============================================

            for obj in objects:

                center = (
                    int(
                        obj.center_x
                    ),
                    int(
                        obj.center_y
                    ),
                )

                # ------------------------------------------
                # SINGLE
                # ------------------------------------------

                single_events = (
                    single_counter.update(
                        track_id=(
                            obj.track_id
                        ),
                        center=center,
                        class_name=(
                            obj.class_name
                        ),
                        confidence=(
                            obj.confidence
                        ),
                        timestamp=(
                            obj.timestamp
                        ),
                        frame_index=(
                            obj.frame
                        ),
                    )
                )

                single_total += save_events(
                    single_events,
                    single_writer,
                    single_class_counts,
                    single_line_counts,
                )

                # ------------------------------------------
                # MULTI
                # ------------------------------------------

                multi_events = (
                    multi_counter.update(
                        track_id=(
                            obj.track_id
                        ),
                        center=center,
                        class_name=(
                            obj.class_name
                        ),
                        confidence=(
                            obj.confidence
                        ),
                        timestamp=(
                            obj.timestamp
                        ),
                        frame_index=(
                            obj.frame
                        ),
                    )
                )

                multi_total += save_events(
                    multi_events,
                    multi_writer,
                    multi_class_counts,
                    multi_line_counts,
                )

                # ==========================================
                # DRAW BBOX
                # ==========================================

                x1 = int(
                    obj.x1
                )

                y1 = int(
                    obj.y1
                )

                x2 = int(
                    obj.x2
                )

                y2 = int(
                    obj.y2
                )

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        y1,
                    ),
                    (
                        x2,
                        y2,
                    ),
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                )

                cv2.circle(
                    frame,
                    center,
                    4,
                    (
                        0,
                        0,
                        255,
                    ),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"{obj.class_name} "
                        f"#{obj.track_id}"
                    ),
                    (
                        x1,
                        max(
                            y1 - 8,
                            20,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (
                        0,
                        255,
                        0,
                    ),
                    1,
                    cv2.LINE_AA,
                )

            # ==============================================
            # DRAW SINGLE LINE
            # ==============================================

            for line in single_lines:

                draw_line(
                    frame,
                    line,
                    (
                        0,
                        255,
                        255,
                    ),
                    "SINGLE: ",
                )

            # ==============================================
            # DRAW MULTI LINES
            # ==============================================

            multi_colors = [
                (
                    0,
                    0,
                    255,
                ),
                (
                    255,
                    0,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]

            for index, line in enumerate(
                multi_lines
            ):

                draw_line(
                    frame,
                    line,
                    multi_colors[
                        index
                        % len(
                            multi_colors
                        )
                    ],
                    "MULTI: ",
                )

            # ==============================================
            # OVERLAY RESULT
            # ==============================================

            cv2.rectangle(
                frame,
                (
                    8,
                    8,
                ),
                (
                    510,
                    145,
                ),
                (
                    0,
                    0,
                    0,
                ),
                -1,
            )

            cv2.putText(
                frame,
                (
                    f"E4 | "
                    f"Frame "
                    f"{frame_index}/"
                    f"{total_frames}"
                ),
                (
                    20,
                    35,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"SINGLE-LINE: "
                    f"{single_total}"
                ),
                (
                    20,
                    75,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"MULTI-LINE: "
                    f"{multi_total}"
                ),
                (
                    20,
                    115,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    255,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # ==============================================
            # WRITE VIDEO
            # ==============================================

            if video_writer is not None:

                video_writer.write(
                    frame
                )

            # ==============================================
            # PREVIEW
            # ==============================================

            if SHOW_PREVIEW:

                preview = (
                    resize_for_preview(
                        frame
                    )
                )

                cv2.imshow(
                    WINDOW_NAME,
                    preview,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    aborted = True
                    break

            # Console progress
            if frame_index % 250 == 0:

                print(
                    f"Frame "
                    f"{frame_index}/"
                    f"{total_frames} | "
                    f"single={single_total} | "
                    f"multi={multi_total}"
                )

    finally:

        cap.release()

        single_file.close()

        multi_file.close()

        if video_writer is not None:

            video_writer.release()

        cv2.destroyAllWindows()

        del tracker

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    if aborted:

        raise RuntimeError(
            "Experiment bi dung bang Q. "
            "Khong tao evaluation tu run chua hoan tat."
        )

    return {
        "single_total": (
            single_total
        ),
        "multi_total": (
            multi_total
        ),
        "single_class_counts": dict(
            single_class_counts
        ),
        "multi_class_counts": dict(
            multi_class_counts
        ),
        "single_line_counts": dict(
            single_line_counts
        ),
        "multi_line_counts": dict(
            multi_line_counts
        ),
    }


# ==========================================================
# STATISTICS + EVALUATION
# ==========================================================

def evaluate_strategy(
    strategy: str,
    events_path: Path,
    gt_counts: dict[str, int],
    output_dir: Path,
):

    statistics_dir = (
        output_dir
        / "statistics"
    )

    charts_dir = (
        output_dir
        / "charts"
    )

    evaluation_dir = (
        output_dir
        / "evaluation"
    )

    statistics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    charts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # STATISTICS — KIET
    # ======================================================

    (
        statistics,
        statistic_paths,
    ) = run_statistics(
        events_path,
        output_dir=(
            statistics_dir
        ),
        interval_seconds=10,
        chart_dir=charts_dir,
        create_charts=True,
    )

    by_class_path = (
        statistic_paths[
            "by_class"
        ]
    )

    # ======================================================
    # GT SNAPSHOT
    # ======================================================

    gt_path = (
        output_dir
        / f"{strategy}_ground_truth.csv"
    )

    save_gt_snapshot(
        gt_path,
        gt_counts,
    )

    # ======================================================
    # BY-CLASS EVALUATION ONLY
    #
    # Ta manual E4 theo total/class,
    # chưa manual theo từng 10 giây.
    #
    # Copy statistics sang tên khác để evaluator
    # không tự tìm *_statistics_by_minute.csv.
    # ======================================================

    evaluation_input = (
        evaluation_dir
        / f"{strategy}_evaluation_input.csv"
    )

    shutil.copyfile(
        by_class_path,
        evaluation_input,
    )

    # ======================================================
    # EVALUATION — KIET
    # ======================================================

    (
        frames,
        paths,
    ) = run_evaluation(
        ground_truth_path=(
            gt_path
        ),
        system_statistics_path=(
            evaluation_input
        ),
        system_time_path=None,
        output_dir=(
            evaluation_dir
        ),
        experiment=(
            f"e4_eval02_{strategy}"
        ),
    )

    summary = (
        frames[
            "summary"
        ].iloc[0]
    )

    return {
        "strategy": strategy,
        "number_of_lines": (
            1
            if strategy == "single_line"
            else 2
        ),
        "gt_total": int(
            summary[
                "manual_total"
            ]
        ),
        "pred_total": int(
            summary[
                "system_total"
            ]
        ),
        "absolute_error": int(
            summary[
                "absolute_error"
            ]
        ),
        "counting_accuracy_pct": float(
            summary[
                "counting_accuracy_pct"
            ]
        ),
        "class_mae": float(
            summary[
                "mean_class_absolute_error"
            ]
        ),
        "macro_class_accuracy_pct": float(
            summary[
                "macro_class_accuracy_pct"
            ]
        ),
        "events_csv": str(
            events_path
        ),
        "statistics_csv": str(
            by_class_path
        ),
    }


# ==========================================================
# LINE-LEVEL EVALUATION
# ==========================================================

def line_evaluation_rows(
    events_path: Path,
    strategy: str,
    gt_by_line: dict[
        str,
        dict[str, int]
    ],
):

    events = pd.read_csv(
        events_path
    )

    rows = []

    for (
        line_name,
        gt_counts,
    ) in gt_by_line.items():

        line_events = events[
            events["line"]
            == line_name
        ]

        pred_counts = (
            line_events[
                "class"
            ]
            .value_counts()
            .to_dict()
        )

        gt_total = total_from_counts(
            gt_counts
        )

        pred_total = int(
            len(
                line_events
            )
        )

        absolute_error = abs(
            pred_total
            - gt_total
        )

        class_errors = []

        row = {
            "strategy": strategy,
            "line": line_name,
            "gt_total": gt_total,
            "pred_total": pred_total,
            "absolute_error": (
                absolute_error
            ),
            "accuracy_pct": (
                counting_accuracy(
                    gt_total,
                    pred_total,
                )
            ),
        }

        for cls in VEHICLE_CLASSES:

            gt_value = int(
                gt_counts.get(
                    cls,
                    0,
                )
            )

            pred_value = int(
                pred_counts.get(
                    cls,
                    0,
                )
            )

            class_errors.append(
                abs(
                    pred_value
                    - gt_value
                )
            )

            row[
                f"gt_{cls}"
            ] = gt_value

            row[
                f"pred_{cls}"
            ] = pred_value

        row[
            "class_mae"
        ] = (
            sum(
                class_errors
            )
            / len(
                VEHICLE_CLASSES
            )
        )

        rows.append(
            row
        )

    return rows


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # VALIDATION
    # ======================================================

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay video:\n"
            f"{VIDEO_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model:\n"
            f"{MODEL_PATH}"
        )

    single_lines = load_lines(
        SINGLE_CONFIG_PATH
    )

    multi_lines = load_lines(
        MULTI_CONFIG_PATH
    )

    if len(
        single_lines
    ) != 1:

        raise ValueError(
            "E4 single-line config "
            "phai co dung 1 line."
        )

    if len(
        multi_lines
    ) != 2:

        raise ValueError(
            "E4 multi-line config "
            "phai co dung 2 line."
        )

    multi_names = {
        line.name
        for line in multi_lines
    }

    expected_multi_names = set(
        MULTI_GT_BY_LINE.keys()
    )

    if (
        multi_names
        != expected_multi_names
    ):

        raise ValueError(
            "\nTen line trong config multi "
            "khong khop Ground Truth.\n"
            f"Config: {sorted(multi_names)}\n"
            f"GT: {sorted(expected_multi_names)}"
        )

    # ======================================================
    # CHECK GT TOTAL
    # ======================================================

    single_gt_total = (
        total_from_counts(
            SINGLE_GT
        )
    )

    multi_gt_total = (
        total_from_counts(
            MULTI_GT
        )
    )

    print("\n" + "=" * 70)
    print("GROUND TRUTH CHECK")
    print("=" * 70)

    print(
        "Single GT:",
        SINGLE_GT,
    )

    print(
        "Single GT Total:",
        single_gt_total,
    )

    print()

    print(
        "Multi GT by line:"
    )

    for (
        line_name,
        counts,
    ) in (
        MULTI_GT_BY_LINE.items()
    ):

        print(
            f"  {line_name}: "
            f"{counts} "
            f"| total="
            f"{total_from_counts(counts)}"
        )

    print(
        "Multi GT Total:",
        multi_gt_total,
    )

    if (
        single_gt_total
        != multi_gt_total
    ):

        raise ValueError(
            "Single GT va Multi GT "
            "khong cung target population."
        )

    # Save GT snapshot
    gt_by_line_path = (
        save_gt_by_line_snapshot()
    )

    # ======================================================
    # RUN VIDEO ONCE
    # ======================================================

    pipeline_result = (
        run_e4_pipeline(
            single_lines,
            multi_lines,
        )
    )

    # ======================================================
    # EVALUATE SINGLE
    # ======================================================

    single_result = (
        evaluate_strategy(
            strategy=(
                "single_line"
            ),
            events_path=(
                SINGLE_EVENTS_PATH
            ),
            gt_counts=(
                SINGLE_GT
            ),
            output_dir=(
                SINGLE_OUTPUT_DIR
            ),
        )
    )

    # ======================================================
    # EVALUATE MULTI
    # ======================================================

    multi_result = (
        evaluate_strategy(
            strategy=(
                "multi_line"
            ),
            events_path=(
                MULTI_EVENTS_PATH
            ),
            gt_counts=(
                MULTI_GT
            ),
            output_dir=(
                MULTI_OUTPUT_DIR
            ),
        )
    )

    # ======================================================
    # STRATEGY COMPARISON
    # ======================================================

    comparison = pd.DataFrame(
        [
            single_result,
            multi_result,
        ]
    )

    comparison_path = (
        OUTPUT_ROOT
        / "e4_counting_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================
    # LINE-LEVEL COMPARISON
    # ======================================================

    single_gt_by_line = {
        single_lines[
            0
        ].name: SINGLE_GT
    }

    line_rows = []

    line_rows.extend(
        line_evaluation_rows(
            SINGLE_EVENTS_PATH,
            "single_line",
            single_gt_by_line,
        )
    )

    line_rows.extend(
        line_evaluation_rows(
            MULTI_EVENTS_PATH,
            "multi_line",
            MULTI_GT_BY_LINE,
        )
    )

    line_comparison = (
        pd.DataFrame(
            line_rows
        )
    )

    line_comparison_path = (
        OUTPUT_ROOT
        / "e4_line_comparison.csv"
    )

    line_comparison.to_csv(
        line_comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================
    # JSON SUMMARY
    # ======================================================

    json_path = (
        OUTPUT_ROOT
        / "e4_counting_comparison.json"
    )

    json_data = {
        "experiment": "E4",
        "video": str(
            VIDEO_PATH
        ),
        "model": str(
            MODEL_PATH
        ),
        "tracker": (
            TRACKER_TYPE
        ),
        "confidence": (
            CONF_THRESHOLD
        ),
        "iou": (
            IOU_THRESHOLD
        ),
        "image_size": (
            IMAGE_SIZE
        ),
        "allowed_direction": (
            ALLOWED_DIRECTION
        ),
        "ground_truth_total": (
            single_gt_total
        ),
        "single_line": (
            single_result
        ),
        "multi_line": (
            multi_result
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ======================================================
    # PRINT FINAL
    # ======================================================

    print("\n" + "=" * 80)
    print("E4 FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Strategy':15s}"
        f"{'GT':>8s}"
        f"{'Pred':>8s}"
        f"{'AE':>8s}"
        f"{'Accuracy':>14s}"
        f"{'Class MAE':>12s}"
    )

    print("-" * 80)

    for result in [
        single_result,
        multi_result,
    ]:

        print(
            f"{result['strategy']:15s}"
            f"{result['gt_total']:>8d}"
            f"{result['pred_total']:>8d}"
            f"{result['absolute_error']:>8d}"
            f"{result['counting_accuracy_pct']:>13.2f}%"
            f"{result['class_mae']:>12.2f}"
        )

    print("\n" + "=" * 80)
    print("MULTI-LINE — BY LINE")
    print("=" * 80)

    for (
        line_name,
        gt_counts,
    ) in (
        MULTI_GT_BY_LINE.items()
    ):

        match = (
            line_comparison[
                (
                    line_comparison[
                        "strategy"
                    ]
                    == "multi_line"
                )
                &
                (
                    line_comparison[
                        "line"
                    ]
                    == line_name
                )
            ]
        )

        if match.empty:
            continue

        row = (
            match.iloc[0]
        )

        print(
            f"{line_name:15s} | "
            f"GT={int(row['gt_total']):3d} | "
            f"Pred={int(row['pred_total']):3d} | "
            f"AE={int(row['absolute_error']):3d} | "
            f"Acc={row['accuracy_pct']:.2f}%"
        )

    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        "Run directory:",
        OUTPUT_ROOT,
    )

    print(
        "Strategy comparison:",
        comparison_path,
    )

    print(
        "Line comparison:",
        line_comparison_path,
    )

    print(
        "GT by line:",
        gt_by_line_path,
    )

    print(
        "Single events:",
        SINGLE_EVENTS_PATH,
    )

    print(
        "Multi events:",
        MULTI_EVENTS_PATH,
    )

    if SAVE_OUTPUT_VIDEO:

        print(
            "Comparison video:",
            OUTPUT_VIDEO_PATH,
        )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:

        print(
            "\nERROR:",
            error,
        )

        raise