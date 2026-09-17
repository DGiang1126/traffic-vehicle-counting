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

LINE_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "line_configs"
    / "eval_02_lines.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "eval_02_manual_counts_by_10s.csv"
)

OUTPUT_CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_PATH = (
    OUTPUT_CSV_DIR
    / "integration_eval02_events.csv"
)


# ==========================================================
# EXPERIMENT SETTINGS
# ==========================================================

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640

TRACKER_TYPE = "bytetrack"

# GT của Kiệt:
# nhìn START -> END, chỉ đếm LEFT -> RIGHT.
ALLOWED_DIRECTION = "negative_to_positive"


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
            start=tuple(item["start"]),
            end=tuple(item["end"]),

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
# MAIN PIPELINE
# ==========================================================

def main():

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

    # đưa model lên GPU
    tracker.model.to(DEVICE)

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

    total_events = 0

    frame_index = 0

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
                # DRAW BBOX
                # ----------------------------------------------

                x1 = int(obj.x1)
                y1 = int(obj.y1)
                x2 = int(obj.x2)
                y2 = int(obj.y2)

                center = (
                    int(obj.center_x),
                    int(obj.center_y),
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    center,
                    4,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"{obj.class_name} #{obj.track_id}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
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
                # SAVE COUNTING EVENTS
                # ==================================================

                for event in new_events:

                    writer.writerow(
                        event.to_dict()
                    )

                    class_counts[
                        event.class_name
                    ] += 1

                    total_events += 1

                    print(
                        f"COUNTED | "
                        f"{event.class_name} | "
                        f"ID={event.track_id} | "
                        f"Line={event.line} | "
                        f"Direction={event.direction} | "
                        f"Total={total_events}"
                    )

            # ==================================================
            # DRAW COUNTING LINES
            # ==================================================

            for line in lines:

                cv2.line(
                    frame,
                    line.start,
                    line.end,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    line.name,
                    line.start,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # ==================================================
            # DISPLAY STATISTICS
            # ==================================================

            y_offset = 30

            cv2.putText(
                frame,
                f"Total: {total_events}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            for cls_name in [
                "car",
                "motorcycle",
                "bus",
                "truck",
                "bicycle",
            ]:

                y_offset += 25

                cv2.putText(
                    frame,
                    (
                        f"{cls_name}: "
                        f"{class_counts[cls_name]}"
                    ),
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                (
                    f"Frame: "
                    f"{frame_index}/"
                    f"{total_frames}"
                ),
                (10, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # ==================================================
            # SHOW
            # ==================================================

            cv2.imshow(
                "Integrated Traffic Counting - Q to quit",
                frame,
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
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
    # STATISTICS — KIỆT
    # ======================================================

    print("\nRunning statistics...")

    statistics, statistic_paths = run_statistics(
        EVENTS_PATH,

        # GT Kiệt eval_02 được chia 10 giây
        interval_seconds=10,

        create_charts=True,
    )

    statistics_path = statistic_paths[
        "by_class"
    ]

    system_by_time_path = statistic_paths[
        "by_minute"
    ]

    print(
        "Statistics:",
        statistics_path,
    )

    # ======================================================
    # EVALUATION — KIỆT
    # ======================================================

    print("\nRunning evaluation...")

    evaluation_frames, evaluation_paths = (
        run_evaluation(
            ground_truth_path=GROUND_TRUTH_PATH,
            system_statistics_path=statistics_path,
            system_time_path=system_by_time_path,
            experiment="integration_eval02",
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
        int(summary["manual_total"]),
    )

    print(
        "Pred Total:",
        int(summary["system_total"]),
    )

    print(
        "AE:",
        int(summary["absolute_error"]),
    )

    print(
        "Counting Accuracy:",
        f"{summary['counting_accuracy_pct']:.2f}%",
    )

    print(
        "Class MAE:",
        f"{summary['mean_class_absolute_error']:.2f}",
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()