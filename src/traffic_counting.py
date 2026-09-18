from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch

from src.tracking.tracker import VehicleTracker
from src.counting import LineDefinition, MultiLineCounter
from src.statistics.statistics import run_statistics
from evaluation.evaluate_counting import run_evaluation


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------
# VIDEO E4: eval_02
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# 2 line chính thức cho E4 eval_02
# ----------------------------------------------------------

LINE_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "line_configs"
    / "eval_02_e4_lines.json"
)

# ----------------------------------------------------------
# GT E4
#
# LƯU Ý:
# GT eval_02 cũ của Kiệt KHÔNG dùng ở đây vì line khác.
#
# Sau khi manual count đúng 2 line E4,
# tạo file này rồi code sẽ tự chạy evaluation.
# ----------------------------------------------------------

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "eval_02_e4_manual_counts_by_10s.csv"
)

OUTPUT_CSV_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "csv"
)

OUTPUT_CSV_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVENTS_PATH = (
    OUTPUT_CSV_DIR
    / "integration_eval02_e4_multiline_events.csv"
)


# ==========================================================
# EXPERIMENT SETTINGS
# ==========================================================

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640

TRACKER_TYPE = "bytetrack"

# ----------------------------------------------------------
# Counting direction
#
# Giữ logic hiện tại của integration.
# Sau khi chốt GT E4 thì không đổi rule nữa.
# ----------------------------------------------------------

ALLOWED_DIRECTION = "negative_to_positive"


# ==========================================================
# PREVIEW SETTINGS
# ==========================================================

# Chỉ resize cửa sổ preview.
# KHÔNG resize frame trước khi tracking/counting.
#
# Như vậy:
# - model vẫn chạy trên frame gốc
# - tọa độ line vẫn đúng
# - preview vừa màn hình

PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720

WINDOW_NAME = (
    "E4 eval_02 - Multi-line Traffic Counting - Q to quit"
)


# ==========================================================
# DEVICE
# ==========================================================

if torch.cuda.is_available():

    DEVICE = "cuda:0"

    print("CUDA available: True")
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

else:

    DEVICE = "cpu"

    print("CUDA available: False")
    print("Dang chay bang CPU")


# ==========================================================
# LOAD COUNTING LINES
# ==========================================================

def load_counting_lines(
    config_path: Path,
) -> list[LineDefinition]:

    if not config_path.exists():

        raise FileNotFoundError(
            "\nKhong tim thay file line config:\n"
            f"{config_path}\n\n"
            "Hay tao file eval_02_e4_lines.json truoc."
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    lines = []

    for item in config["lines"]:

        line = LineDefinition(
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

        lines.append(line)

    return lines


# ==========================================================
# RESIZE PREVIEW
# ==========================================================

def resize_for_preview(
    frame,
    max_width: int = PREVIEW_MAX_WIDTH,
    max_height: int = PREVIEW_MAX_HEIGHT,
):

    height, width = frame.shape[:2]

    scale_width = (
        max_width / width
    )

    scale_height = (
        max_height / height
    )

    scale = min(
        scale_width,
        scale_height,
        1.0,
    )

    # Nếu frame đã nhỏ hơn giới hạn
    # thì giữ nguyên.
    if scale >= 1.0:
        return frame

    preview_width = int(
        width * scale
    )

    preview_height = int(
        height * scale
    )

    preview = cv2.resize(
        frame,
        (
            preview_width,
            preview_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return preview


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    print("\n" + "=" * 60)
    print("E4 - EVAL_02 MULTI-LINE INTEGRATION")
    print("=" * 60)

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

    # ======================================================
    # TRACKER — THUẬN
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
    # COUNTING — ĐỨC ANH
    # ======================================================

    lines = load_counting_lines(
        LINE_CONFIG_PATH
    )

    counter = MultiLineCounter(
        lines,
        allowed_direction=ALLOWED_DIRECTION,
    )

    print("\nCounting lines:")

    for line in lines:

        print(
            f"{line.name}: "
            f"{line.start} -> {line.end}"
        )

    print(
        "\nAllowed direction:",
        ALLOWED_DIRECTION,
    )

    # ======================================================
    # OPEN VIDEO
    # ======================================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Khong mo duoc video: {VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

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

    print(
        "\nVideo resolution:",
        f"{frame_width}x{frame_height}",
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
    # EVENT CSV
    # ======================================================

    event_file = open(
        EVENTS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
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

    writer.writeheader()

    # ======================================================
    # COUNTERS FOR DISPLAY
    # ======================================================

    class_counts = defaultdict(int)

    line_counts = defaultdict(int)

    direction_counts = defaultdict(int)

    total_events = 0

    frame_index = 0

    # ======================================================
    # WINDOW
    # ======================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    # ======================================================
    # PROCESS VIDEO
    # ======================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            # ==================================================
            # TRACKING — THUẬN
            # ==================================================

            objects = tracker.track_frame(
                frame,
                frame_number=frame_index,
                fps=fps,
            )

            # ==================================================
            # VEHICLES
            # ==================================================

            for obj in objects:

                # ----------------------------------------------
                # BBOX
                # ----------------------------------------------

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

                # ----------------------------------------------
                # DRAW BBOX
                # ----------------------------------------------

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

                # ----------------------------------------------
                # DRAW CENTER
                # ----------------------------------------------

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

                # ----------------------------------------------
                # TRACK LABEL
                # ----------------------------------------------

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
                    0.5,
                    (
                        0,
                        255,
                        0,
                    ),
                    1,
                    cv2.LINE_AA,
                )

                # ==================================================
                # COUNTING — ĐỨC ANH
                # ==================================================

                new_events = counter.update(
                    track_id=obj.track_id,
                    center=center,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    timestamp=obj.timestamp,
                    frame_index=obj.frame,
                )

                # ==================================================
                # SAVE EVENTS
                # ==================================================

                for event in new_events:

                    writer.writerow(
                        event.to_dict()
                    )

                    class_counts[
                        event.class_name
                    ] += 1

                    line_counts[
                        event.line
                    ] += 1

                    direction_counts[
                        event.direction
                    ] += 1

                    total_events += 1

                    print(
                        "COUNTED | "
                        f"{event.class_name} | "
                        f"ID={event.track_id} | "
                        f"Line={event.line} | "
                        f"Direction={event.direction} | "
                        f"Total={total_events}"
                    )

            # ==================================================
            # DRAW COUNTING LINES
            # ==================================================

            for line_index, line in enumerate(
                lines
            ):

                # Mỗi line dùng style khác một chút
                # để dễ nhìn trên preview.

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

                # ----------------------------------------------
                # DRAW LINE
                # ----------------------------------------------

                cv2.line(
                    frame,
                    line.start,
                    line.end,
                    line_color,
                    5,
                    cv2.LINE_AA,
                )

                # ----------------------------------------------
                # DRAW ENDPOINTS
                # ----------------------------------------------

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

                # ----------------------------------------------
                # LINE NAME
                # ----------------------------------------------

                label_x = (
                    line.start[0] + 10
                )

                label_y = (
                    line.start[1] - 10
                )

                label_y = max(
                    label_y,
                    25,
                )

                cv2.putText(
                    frame,
                    line.name,
                    (
                        label_x,
                        label_y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    line_color,
                    3,
                    cv2.LINE_AA,
                )

            # ==================================================
            # DISPLAY STATISTICS
            # ==================================================

            y_offset = 35

            # ----------------------------------------------
            # TOTAL
            # ----------------------------------------------

            cv2.putText(
                frame,
                f"Total: {total_events}",
                (
                    15,
                    y_offset,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    255,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            # ----------------------------------------------
            # CLASS COUNTS
            # ----------------------------------------------

            for cls_name in [
                "car",
                "motorcycle",
                "bus",
                "truck",
                "bicycle",
            ]:

                y_offset += 28

                cv2.putText(
                    frame,
                    (
                        f"{cls_name}: "
                        f"{class_counts[cls_name]}"
                    ),
                    (
                        15,
                        y_offset,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (
                        255,
                        255,
                        0,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            # ----------------------------------------------
            # COUNTS BY LINE
            # ----------------------------------------------

            y_offset += 35

            cv2.putText(
                frame,
                "By line:",
                (
                    15,
                    y_offset,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            for line in lines:

                y_offset += 27

                cv2.putText(
                    frame,
                    (
                        f"{line.name}: "
                        f"{line_counts[line.name]}"
                    ),
                    (
                        15,
                        y_offset,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            # ----------------------------------------------
            # FRAME
            # ----------------------------------------------

            cv2.putText(
                frame,
                (
                    f"Frame: "
                    f"{frame_index}/"
                    f"{total_frames}"
                ),
                (
                    15,
                    frame_height - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # ==================================================
            # PREVIEW RESIZE
            # ==================================================

            preview = resize_for_preview(
                frame
            )

            cv2.imshow(
                WINDOW_NAME,
                preview,
            )

            # ==================================================
            # QUIT
            # ==================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                print(
                    "\nUser pressed Q. Stop."
                )
                break

    finally:

        cap.release()

        event_file.close()

        cv2.destroyAllWindows()

    # ======================================================
    # FINAL SYSTEM COUNT
    # ======================================================

    print("\n" + "=" * 60)
    print("INTEGRATION RESULT")
    print("=" * 60)

    for cls_name in [
        "car",
        "motorcycle",
        "bus",
        "truck",
        "bicycle",
    ]:

        print(
            f"{cls_name:12s}: "
            f"{class_counts[cls_name]}"
        )

    print("-" * 60)

    print(
        "Total:",
        total_events,
    )

    print(
        "Events CSV:",
        EVENTS_PATH,
    )

    # ======================================================
    # RESULT BY LINE
    # ======================================================

    print("\n" + "=" * 60)
    print("COUNT BY LINE")
    print("=" * 60)

    for line in lines:

        print(
            f"{line.name:15s}: "
            f"{line_counts[line.name]}"
        )

    # ======================================================
    # RESULT BY DIRECTION
    # ======================================================

    print("\n" + "=" * 60)
    print("COUNT BY DIRECTION")
    print("=" * 60)

    if direction_counts:

        for direction, count in (
            direction_counts.items()
        ):

            print(
                f"{direction:25s}: "
                f"{count}"
            )

    else:

        print(
            "No direction events."
        )

    # ======================================================
    # STATISTICS — KIỆT
    # ======================================================

    print("\nRunning statistics...")

    statistics, statistic_paths = (
        run_statistics(
            EVENTS_PATH,
            interval_seconds=10,
            create_charts=True,
        )
    )

    statistics_path = (
        statistic_paths[
            "by_class"
        ]
    )

    system_by_time_path = (
        statistic_paths[
            "by_minute"
        ]
    )

    print(
        "Statistics:",
        statistics_path,
    )

    # ======================================================
    # EVALUATION — KIỆT
    # ======================================================
    #
    # Chỉ chạy khi đã có GT được manual count
    # đúng theo 2 line E4 này.
    #
    # Không sử dụng GT eval_02 cũ của Kiệt
    # vì line cũ khác line E4.
    # ======================================================

    if not GROUND_TRUTH_PATH.exists():

        print("\n" + "=" * 60)
        print("EVALUATION SKIPPED")
        print("=" * 60)

        print(
            "Chua co matching Ground Truth "
            "cho 2 line E4."
        )

        print(
            "Can tao file:"
        )

        print(
            GROUND_TRUTH_PATH
        )

        print(
            "\nKHONG dung GT eval_02 cu cua Kiet "
            "de cham 2 line nay."
        )

        return

    # ======================================================
    # RUN EVALUATION
    # ======================================================

    print("\nRunning evaluation...")

    evaluation_frames, evaluation_paths = (
        run_evaluation(
            ground_truth_path=(
                GROUND_TRUTH_PATH
            ),

            system_statistics_path=(
                statistics_path
            ),

            system_time_path=(
                system_by_time_path
            ),

            experiment=(
                "integration_eval02_e4_multiline"
            ),
        )
    )

    summary = (
        evaluation_frames[
            "summary"
        ].iloc[0]
    )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    print(
        "GT Total:",
        int(
            summary[
                "manual_total"
            ]
        ),
    )

    print(
        "Pred Total:",
        int(
            summary[
                "system_total"
            ]
        ),
    )

    print(
        "AE:",
        int(
            summary[
                "absolute_error"
            ]
        ),
    )

    print(
        "Counting Accuracy:",
        (
            f"{summary['counting_accuracy_pct']:.2f}%"
        ),
    )

    print(
        "Class MAE:",
        (
            f"{summary['mean_class_absolute_error']:.2f}"
        ),
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()