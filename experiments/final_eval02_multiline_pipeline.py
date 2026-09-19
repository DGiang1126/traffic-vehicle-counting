from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================
# Khi chạy:
#   python .\experiments\final_eval02_multiline_pipeline.py
# Python sẽ lấy thư mục experiments/ làm sys.path[0].
# Vì vậy cần thêm thư mục root của project để import được src/ và evaluation/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from src.tracking.tracker import VehicleTracker
from src.counting import LineDefinition, MultiLineCounter
from src.statistics.statistics import run_statistics
from evaluation.evaluate_counting import run_evaluation


# ============================================================
# PROJECT PATHS
# ============================================================

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

# FINAL DEMO: dùng 2 line đã chốt trong E4.
LINE_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "line_configs"
    / "eval_02_e4_multi_line.json"
)


# ============================================================
# FINAL PIPELINE SETTINGS
# ============================================================

TRACKER_TYPE = "bytetrack"
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640
ALLOWED_DIRECTION = "negative_to_positive"

VEHICLE_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
]

SHOW_PREVIEW = True
WINDOW_NAME = "Final eval_02 MULTI-LINE pipeline - Q to stop"


# ============================================================
# MANUAL GROUND TRUTH FOR THE TWO FINAL E4 LINES
# ============================================================
#
# Đây là GT của đúng 2 line:
#   Cong_chinh + Nhanh_phu
#
# Không dùng eval_02_manual_counts_by_10s.csv ở đây vì file đó
# thuộc counting line cũ (single line), không cùng hình học với
# 2 line final.
#
# GT này dùng cho:
#   - evaluation tổng toàn video
#   - evaluation từng line
#
# Chưa có manual GT theo từng 10 giây cho 2 line này, nên:
#   - statistics_10s vẫn được xuất để xem xu hướng
#   - KHÔNG tính Time MAE cho multi-line
# ============================================================

MULTI_GT_BY_LINE = {
    "Line 1": {
        "car": 161,
        "motorcycle": 2,
        "bus": 2,
        "truck": 15,
        "bicycle": 0,
    },
    "Line 2": {
        "car": 17,
        "motorcycle": 0,
        "bus": 0,
        "truck": 2,
        "bicycle": 0,
    },
}


def aggregate_gt() -> dict[str, int]:
    result = {
        cls_name: 0
        for cls_name in VEHICLE_CLASSES
    }

    for line_counts in MULTI_GT_BY_LINE.values():
        for cls_name in VEHICLE_CLASSES:
            result[cls_name] += int(
                line_counts.get(cls_name, 0)
            )

    return result


MULTI_GT = aggregate_gt()


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_eval02_multiline_runs"
    / RUN_ID
)

VIDEO_OUTPUT_DIR = OUTPUT_ROOT / "video"
EVENTS_OUTPUT_DIR = OUTPUT_ROOT / "events"

STATISTICS_10S_DIR = OUTPUT_ROOT / "statistics_10s"
STATISTICS_60S_DIR = OUTPUT_ROOT / "statistics_60s"

CHARTS_10S_DIR = OUTPUT_ROOT / "charts_10s"
CHARTS_60S_DIR = OUTPUT_ROOT / "charts_60s"

EVALUATION_DIR = OUTPUT_ROOT / "evaluation"
INPUT_SNAPSHOT_DIR = OUTPUT_ROOT / "input_snapshot"

for directory in [
    VIDEO_OUTPUT_DIR,
    EVENTS_OUTPUT_DIR,
    STATISTICS_10S_DIR,
    STATISTICS_60S_DIR,
    CHARTS_10S_DIR,
    CHARTS_60S_DIR,
    EVALUATION_DIR,
    INPUT_SNAPSHOT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

OUTPUT_VIDEO_PATH = (
    VIDEO_OUTPUT_DIR
    / "final_eval02_multiline_demo.mp4"
)

EVENTS_PATH = (
    EVENTS_OUTPUT_DIR
    / "final_eval02_multiline_events.csv"
)

GT_TOTAL_PATH = (
    INPUT_SNAPSHOT_DIR
    / "eval02_multiline_gt_total.csv"
)

GT_BY_LINE_PATH = (
    INPUT_SNAPSHOT_DIR
    / "eval02_multiline_gt_by_line.csv"
)

LINE_EVALUATION_PATH = (
    EVALUATION_DIR
    / "final_eval02_multiline_evaluation_by_line.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_counting_lines(
    config_path: Path,
) -> list[LineDefinition]:

    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy line config: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    lines: list[LineDefinition] = []

    for item in config["lines"]:
        lines.append(
            LineDefinition(
                name=item["name"],
                start=tuple(
                    map(
                        int,
                        item["start"],
                    )
                ),
                end=tuple(
                    map(
                        int,
                        item["end"],
                    )
                ),
                negative_to_positive=(
                    "negative_to_positive"
                ),
                positive_to_negative=(
                    "positive_to_negative"
                ),
            )
        )

    return lines


def draw_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: float = 0.6,
    thickness: int = 2,
):
    """Text trắng có viền đen để dễ đọc."""

    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def counting_accuracy(
    manual: int,
    system: int,
) -> float:

    if manual == 0:
        return (
            100.0
            if system == 0
            else 0.0
        )

    return 100.0 * (
        1.0
        - abs(system - manual) / manual
    )


def save_gt_files():
    # --------------------------------------------------------
    # GT TOTAL
    # --------------------------------------------------------
    #
    # evaluate_counting.py yêu cầu GT có minute hoặc
    # start_sec/end_sec. Vì đây là GT toàn video, dùng một
    # khoảng thời gian duy nhất.
    #
    # KHÔNG truyền system_time_path để tránh đánh giá time bins.
    # --------------------------------------------------------

    with open(
        GT_TOTAL_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "start_sec",
                "end_sec",
                *VEHICLE_CLASSES,
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "start_sec": 0.0,
                "end_sec": 179.067,
                **MULTI_GT,
            }
        )

    # --------------------------------------------------------
    # GT BY LINE
    # --------------------------------------------------------

    with open(
        GT_BY_LINE_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "line",
                *VEHICLE_CLASSES,
                "total",
            ],
        )

        writer.writeheader()

        for (
            line_name,
            counts,
        ) in MULTI_GT_BY_LINE.items():

            writer.writerow(
                {
                    "line": line_name,
                    **counts,
                    "total": sum(
                        counts.values()
                    ),
                }
            )


def save_line_evaluation(
    line_class_counts: dict,
):
    rows = []

    for (
        line_name,
        gt_counts,
    ) in MULTI_GT_BY_LINE.items():

        pred_counts = {
            cls_name: int(
                line_class_counts[
                    line_name
                ][
                    cls_name
                ]
            )
            for cls_name
            in VEHICLE_CLASSES
        }

        gt_total = sum(
            gt_counts.values()
        )

        pred_total = sum(
            pred_counts.values()
        )

        row = {
            "line": line_name,
            "gt_total": gt_total,
            "pred_total": pred_total,
            "absolute_error": abs(
                pred_total
                - gt_total
            ),
            "accuracy_pct": round(
                counting_accuracy(
                    gt_total,
                    pred_total,
                ),
                4,
            ),
        }

        class_abs_errors = []

        for cls_name in VEHICLE_CLASSES:
            gt_value = int(
                gt_counts.get(
                    cls_name,
                    0,
                )
            )

            pred_value = int(
                pred_counts.get(
                    cls_name,
                    0,
                )
            )

            row[
                f"gt_{cls_name}"
            ] = gt_value

            row[
                f"pred_{cls_name}"
            ] = pred_value

            class_abs_errors.append(
                abs(
                    pred_value
                    - gt_value
                )
            )

        row["class_mae"] = round(
            sum(class_abs_errors)
            / len(VEHICLE_CLASSES),
            4,
        )

        rows.append(row)

    with open(
        LINE_EVALUATION_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        if not rows:
            return

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_run_readme(
    *,
    fps: float,
    frame_count: int,
    total_events: int,
    class_counts: dict[str, int],
    line_counts: dict[str, int],
    evaluation_summary: dict,
):
    readme_path = (
        OUTPUT_ROOT
        / "README.txt"
    )

    lines = [
        "FINAL EVAL_02 MULTI-LINE PIPELINE RUN",
        "=" * 72,
        "",
        f"Run ID: {RUN_ID}",
        f"Input video: {VIDEO_PATH}",
        f"Model: {MODEL_PATH.name}",
        f"Tracker: {TRACKER_TYPE}",
        f"Confidence threshold: {CONF_THRESHOLD}",
        f"IoU threshold: {IOU_THRESHOLD}",
        f"Image size: {IMAGE_SIZE}",
        f"Counting direction: {ALLOWED_DIRECTION}",
        f"Frames processed: {frame_count}",
        f"Video FPS: {fps:.3f}",
        "",
        "COUNTING LINES",
        "-" * 72,
    ]

    for line_name in MULTI_GT_BY_LINE:
        lines.append(
            f"{line_name}: "
            f"{line_counts.get(line_name, 0)}"
        )

    lines.extend(
        [
            "",
            "FINAL SYSTEM COUNT",
            "-" * 72,
        ]
    )

    for cls_name in VEHICLE_CLASSES:
        lines.append(
            f"{cls_name}: "
            f"{class_counts.get(cls_name, 0)}"
        )

    lines.extend(
        [
            f"TOTAL: {total_events}",
            "",
            "EVALUATION — WHOLE VIDEO",
            "-" * 72,
            (
                "Manual total: "
                f"{evaluation_summary['manual_total']}"
            ),
            (
                "System total: "
                f"{evaluation_summary['system_total']}"
            ),
            (
                "Absolute error: "
                f"{evaluation_summary['absolute_error']}"
            ),
            (
                "Counting accuracy: "
                f"{evaluation_summary['counting_accuracy_pct']:.2f}%"
            ),
            (
                "Mean class absolute error: "
                f"{evaluation_summary['mean_class_absolute_error']:.2f}"
            ),
            "",
            "IMPORTANT",
            "-" * 72,
            (
                "This two-line final demo uses the E4 multi-line "
                "manual Ground Truth."
            ),
            (
                "The original eval_02 10-second manual GT belongs "
                "to the old single counting line, so it is NOT used "
                "for multi-line Time MAE."
            ),
            (
                "statistics_10s is still generated to inspect traffic "
                "flow, but it is system-only for this multi-line run."
            ),
            "",
            "OUTPUT FOLDERS",
            "-" * 72,
            (
                "video/           -> rendered video with bbox, track ID, "
                "two lines and live counts"
            ),
            (
                "events/          -> one row per valid crossing event"
            ),
            (
                "statistics_10s/  -> 10-second system statistics"
            ),
            (
                "statistics_60s/  -> per-minute statistics for report"
            ),
            (
                "charts_10s/      -> detailed traffic charts"
            ),
            (
                "charts_60s/      -> report-ready per-minute charts"
            ),
            (
                "evaluation/      -> whole-video and by-line evaluation"
            ),
            (
                "input_snapshot/  -> exact config and GT used by this run"
            ),
        ]
    )

    readme_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    for required_path in [
        VIDEO_PATH,
        MODEL_PATH,
        LINE_CONFIG_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {required_path}"
            )

    # --------------------------------------------------------
    # SNAPSHOT INPUT
    # --------------------------------------------------------

    shutil.copy2(
        LINE_CONFIG_PATH,
        (
            INPUT_SNAPSHOT_DIR
            / LINE_CONFIG_PATH.name
        ),
    )

    save_gt_files()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = "cuda:0"

        print(
            "CUDA available: True"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    else:
        device = "cpu"

        print(
            "CUDA available: False"
        )

        print(
            "Running on CPU"
        )

    # --------------------------------------------------------
    # TRACKER
    # --------------------------------------------------------

    tracker = VehicleTracker(
        model_path=MODEL_PATH,
        tracker_type=TRACKER_TYPE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
    )

    tracker.model.to(
        device
    )

    # --------------------------------------------------------
    # TWO-LINE COUNTER
    # --------------------------------------------------------

    lines = load_counting_lines(
        LINE_CONFIG_PATH
    )

    if len(lines) != 2:
        raise ValueError(
            "Final multi-line demo phải có đúng 2 counting lines."
        )

    config_names = {
        line.name
        for line in lines
    }

    gt_names = set(
        MULTI_GT_BY_LINE.keys()
    )

    if config_names != gt_names:
        raise ValueError(
            "Tên line trong config không khớp GT.\n"
            f"Config: {sorted(config_names)}\n"
            f"GT: {sorted(gt_names)}"
        )

    counter = MultiLineCounter(
        lines,
        allowed_direction=(
            ALLOWED_DIRECTION
        ),
    )

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được video: {VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    video_writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not video_writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Không tạo được output video: "
            f"{OUTPUT_VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # EVENTS CSV
    # --------------------------------------------------------

    event_file = open(
        EVENTS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )

    event_writer = csv.DictWriter(
        event_file,
        fieldnames=[
            "timestamp",
            "frame_index",
            "track_id",
            "class",
            "line",
            "direction",
            "confidence",
        ],
    )

    event_writer.writeheader()

    # --------------------------------------------------------
    # LIVE STATE
    # --------------------------------------------------------

    class_counts = defaultdict(
        int
    )

    line_counts = defaultdict(
        int
    )

    line_class_counts = defaultdict(
        lambda: defaultdict(int)
    )

    total_events = 0
    frame_index = 0
    aborted = False

    # --------------------------------------------------------
    # PREVIEW WINDOW
    # --------------------------------------------------------

    if SHOW_PREVIEW:
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL EVAL_02 MULTI-LINE PIPELINE"
    )

    print(
        "=" * 72
    )

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
        "Lines:"
    )

    for line in lines:
        print(
            f"  {line.name}: "
            f"{line.start} -> {line.end}"
        )

    print(
        "Output:",
        OUTPUT_ROOT,
    )

    print(
        "=" * 72
        + "\n"
    )

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            frame_index += 1

            # ------------------------------------------------
            # DETECTION + TRACKING
            # ------------------------------------------------

            objects = (
                tracker.track_frame(
                    frame,
                    frame_number=(
                        frame_index
                    ),
                    fps=fps,
                )
            )

            # ------------------------------------------------
            # DRAW + COUNT
            # ------------------------------------------------

            for obj in objects:

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

                center = (
                    int(
                        obj.center_x
                    ),
                    int(
                        obj.center_y
                    ),
                )

                # Bounding box
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

                # Center
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

                # Class + ID + confidence
                draw_text(
                    frame,
                    (
                        f"{obj.class_name} "
                        f"#{obj.track_id} "
                    ),
                    x1,
                    max(
                        y1 - 8,
                        20,
                    ),
                    scale=0.6,
                    thickness=2,
                )

                # Counting
                new_events = (
                    counter.update(
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

                for event in new_events:

                    event_writer.writerow(
                        event.to_dict()
                    )

                    class_counts[
                        event.class_name
                    ] += 1

                    line_counts[
                        event.line
                    ] += 1

                    line_class_counts[
                        event.line
                    ][
                        event.class_name
                    ] += 1

                    total_events += 1

            # ------------------------------------------------
            # DRAW 2 COUNTING LINES
            # ------------------------------------------------

            for (
                line_index,
                line,
            ) in enumerate(lines):

                if line_index == 0:
                    line_color = (
                        0,
                        0,
                        255,
                    )
                else:
                    line_color = (
                        255,
                        0,
                        255,
                    )

                cv2.line(
                    frame,
                    line.start,
                    line.end,
                    line_color,
                    5,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    line.start,
                    8,
                    line_color,
                    -1,
                )

                cv2.circle(
                    frame,
                    line.end,
                    8,
                    line_color,
                    -1,
                )

                draw_text(
                    frame,
                    line.name,
                    (
                        line.start[0]
                        + 10
                    ),
                    max(
                        line.start[1]
                        - 10,
                        25,
                    ),
                    scale=0.75,
                )

            # ------------------------------------------------
            # LIVE INFO PANEL
            # ------------------------------------------------

            panel_height = 330

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (
                    10,
                    10,
                ),
                (
                    420,
                    panel_height,
                ),
                (
                    0,
                    0,
                    0,
                ),
                -1,
            )

            frame = cv2.addWeighted(
                overlay,
                0.55,
                frame,
                0.45,
                0,
            )

            y = 40

            draw_text(
                frame,
                (
                    "FINAL SYSTEM: "
                    "YOLOv8s + ByteTrack"
                ),
                25,
                y,
                scale=0.65,
            )

            y += 38

            draw_text(
                frame,
                (
                    f"Total: "
                    f"{total_events}"
                ),
                25,
                y,
                scale=0.8,
            )

            # Counts by class
            for cls_name in (
                VEHICLE_CLASSES
            ):

                y += 27

                draw_text(
                    frame,
                    (
                        f"{cls_name}: "
                        f"{class_counts[cls_name]}"
                    ),
                    25,
                    y,
                    scale=0.55,
                    thickness=1,
                )

            # Counts by line
            y += 34

            draw_text(
                frame,
                "By line:",
                25,
                y,
                scale=0.58,
            )

            for line in lines:

                y += 27

                draw_text(
                    frame,
                    (
                        f"{line.name}: "
                        f"{line_counts[line.name]}"
                    ),
                    25,
                    y,
                    scale=0.55,
                    thickness=1,
                )

            # Frame progress
            draw_text(
                frame,
                (
                    f"Frame "
                    f"{frame_index}/"
                    f"{total_frames}"
                ),
                25,
                (
                    frame_height
                    - 25
                ),
                scale=0.55,
                thickness=1,
            )

            # ------------------------------------------------
            # SAVE VIDEO
            # ------------------------------------------------

            video_writer.write(
                frame
            )

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            if SHOW_PREVIEW:

                preview = frame

                if frame_width > 1280:

                    scale = (
                        1280
                        / frame_width
                    )

                    preview = (
                        cv2.resize(
                            frame,
                            (
                                int(
                                    frame_width
                                    * scale
                                ),
                                int(
                                    frame_height
                                    * scale
                                ),
                            ),
                            interpolation=(
                                cv2.INTER_AREA
                            ),
                        )
                    )

                cv2.imshow(
                    WINDOW_NAME,
                    preview,
                )

                if (
                    cv2.waitKey(1)
                    & 0xFF
                    == ord("q")
                ):
                    aborted = True
                    break

            if (
                frame_index
                % 250
                == 0
            ):
                print(
                    f"Frame "
                    f"{frame_index}/"
                    f"{total_frames} | "
                    f"total="
                    f"{total_events} | "
                    + " | ".join(
                        (
                            f"{line.name}="
                            f"{line_counts[line.name]}"
                        )
                        for line
                        in lines
                    )
                )

    finally:

        cap.release()

        video_writer.release()

        event_file.close()

        cv2.destroyAllWindows()

        del tracker

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if aborted:
        raise RuntimeError(
            "Run bị dừng bằng Q. "
            "Không tạo statistics/evaluation "
            "từ run chưa hoàn tất."
        )

    # ========================================================
    # STATISTICS — 10 SECONDS
    # ========================================================

    print(
        "\nRunning 10-second statistics..."
    )

    (
        stats_10s,
        stats_10s_paths,
    ) = run_statistics(
        EVENTS_PATH,
        output_dir=(
            STATISTICS_10S_DIR
        ),
        interval_seconds=10,
        chart_dir=(
            CHARTS_10S_DIR
        ),
        create_charts=True,
    )

    # ========================================================
    # STATISTICS — 60 SECONDS
    # ========================================================

    print(
        "\nRunning 60-second statistics..."
    )

    (
        stats_60s,
        stats_60s_paths,
    ) = run_statistics(
        EVENTS_PATH,
        output_dir=(
            STATISTICS_60S_DIR
        ),
        interval_seconds=60,
        chart_dir=(
            CHARTS_60S_DIR
        ),
        create_charts=True,
    )

    # ========================================================
    # WHOLE-VIDEO EVALUATION
    # ========================================================
    #
    # Copy by-class stats to a filename without a sibling
    # *_by_minute.csv. This prevents evaluate_counting.py from
    # auto-loading 10-second bins. We only evaluate aggregate
    # counts because multi-line GT has not been manually labeled
    # by time bin.
    # ========================================================

    print(
        "\nRunning whole-video multi-line evaluation..."
    )

    evaluation_input = (
        EVALUATION_DIR
        / "final_eval02_multiline_evaluation_input.csv"
    )

    shutil.copyfile(
        stats_10s_paths[
            "by_class"
        ],
        evaluation_input,
    )

    (
        evaluation_frames,
        evaluation_paths,
    ) = run_evaluation(
        ground_truth_path=(
            GT_TOTAL_PATH
        ),
        system_statistics_path=(
            evaluation_input
        ),
        system_time_path=None,
        output_dir=(
            EVALUATION_DIR
        ),
        experiment=(
            "final_eval02_multiline"
        ),
    )

    evaluation_summary = (
        evaluation_frames[
            "summary"
        ]
        .iloc[0]
        .to_dict()
    )

    # ========================================================
    # BY-LINE EVALUATION
    # ========================================================

    save_line_evaluation(
        line_class_counts
    )

    # ========================================================
    # RUN SUMMARY JSON
    # ========================================================

    summary_json = {
        "run_id": RUN_ID,
        "input_video": str(
            VIDEO_PATH
        ),
        "model": str(
            MODEL_PATH
        ),
        "tracker": TRACKER_TYPE,
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "imgsz": IMAGE_SIZE,
        "allowed_direction": (
            ALLOWED_DIRECTION
        ),
        "counting_lines": [
            {
                "name": line.name,
                "start": list(
                    line.start
                ),
                "end": list(
                    line.end
                ),
            }
            for line
            in lines
        ],
        "frames_processed": (
            frame_index
        ),
        "video_fps": fps,
        "system_total": (
            total_events
        ),
        "system_by_class": {
            cls_name: int(
                class_counts[
                    cls_name
                ]
            )
            for cls_name
            in VEHICLE_CLASSES
        },
        "system_by_line": {
            line.name: int(
                line_counts[
                    line.name
                ]
            )
            for line
            in lines
        },
        "manual_gt_total": (
            sum(
                MULTI_GT.values()
            )
        ),
        "manual_gt_by_class": (
            MULTI_GT
        ),
        "manual_gt_by_line": (
            MULTI_GT_BY_LINE
        ),
        "evaluation": {
            key: (
                None
                if str(value) == "nan"
                else value
            )
            for (
                key,
                value,
            )
            in (
                evaluation_summary.items()
            )
        },
        "output_video": str(
            OUTPUT_VIDEO_PATH
        ),
        "events_csv": str(
            EVENTS_PATH
        ),
    }

    with open(
        (
            OUTPUT_ROOT
            / "run_summary.json"
        ),
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_json,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # README
    # ========================================================

    save_run_readme(
        fps=fps,
        frame_count=(
            frame_index
        ),
        total_events=(
            total_events
        ),
        class_counts=dict(
            class_counts
        ),
        line_counts=dict(
            line_counts
        ),
        evaluation_summary=(
            evaluation_summary
        ),
    )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL EVAL_02 MULTI-LINE RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "Manual total:",
        int(
            evaluation_summary[
                "manual_total"
            ]
        ),
    )

    print(
        "System total:",
        int(
            evaluation_summary[
                "system_total"
            ]
        ),
    )

    print(
        "Absolute error:",
        int(
            evaluation_summary[
                "absolute_error"
            ]
        ),
    )

    print(
        "Counting accuracy:",
        (
            f"{evaluation_summary['counting_accuracy_pct']:.2f}%"
        ),
    )

    print(
        "\nBy line:"
    )

    for line in lines:

        gt_total = sum(
            MULTI_GT_BY_LINE[
                line.name
            ].values()
        )

        pred_total = int(
            line_counts[
                line.name
            ]
        )

        print(
            f"  {line.name}: "
            f"GT={gt_total} | "
            f"Pred={pred_total} | "
            f"AE="
            f"{abs(pred_total - gt_total)} | "
            f"Acc="
            f"{counting_accuracy(gt_total, pred_total):.2f}%"
        )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nImportant files:"
    )

    print(
        "Video:",
        OUTPUT_VIDEO_PATH,
    )

    print(
        "Events:",
        EVENTS_PATH,
    )

    print(
        "60s class chart:",
        stats_60s_paths[
            "chart_class_counts"
        ],
    )

    print(
        "60s traffic-flow chart:",
        stats_60s_paths[
            "chart_traffic_flow"
        ],
    )

    print(
        "Evaluation summary:",
        evaluation_paths[
            "summary"
        ],
    )

    print(
        "By-line evaluation:",
        LINE_EVALUATION_PATH,
    )


if __name__ == "__main__":

    try:
        main()

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:

        print(
            "\nERROR:",
            error,
        )

        raise
