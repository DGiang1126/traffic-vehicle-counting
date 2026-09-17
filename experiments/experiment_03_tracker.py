from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import sys
import time

import cv2
import matplotlib.pyplot as plt
import torch


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORT TRACKER CUA THUAN
# ==========================================================

from src.tracking.tracker import VehicleTracker


# ==========================================================
# EXPERIMENT CONFIG
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

TRACKERS = {
    "ByteTrack": "bytetrack",
    "BoT-SORT": "botsort",
}

CONF_THRESHOLD = 0.2

IOU_THRESHOLD = 0.7

IMAGE_SIZE = 640


# ==========================================================
# OFFICIAL COUNTING LINE - EVAL 02
# ==========================================================

LINE_NAME = "line_1"

LINE_START = (66, 811)
LINE_END = (1744, 801)

# Official rule:
#
# Nhin START -> END
# count LEFT -> RIGHT
#
# Trong toa do anh:
# LEFT  = negative side
# RIGHT = positive side
#
# => count negative -> positive


# ==========================================================
# VEHICLE CLASSES
# ==========================================================

VEHICLE_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
]


# ==========================================================
# OFFICIAL GROUND TRUTH - EVAL 02
# ==========================================================

GROUND_TRUTH = {
    "car": 178,
    "motorcycle": 2,
    "bus": 2,
    "truck": 17,
    "bicycle": 0,
}

GROUND_TRUTH_TOTAL = sum(
    GROUND_TRUTH.values()
)


# ==========================================================
# OUTPUT PATHS
# ==========================================================

CSV_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "csv"
)

CHART_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)

CSV_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHART_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV_PATH = (
    CSV_OUTPUT_DIR
    / "e3_tracker_comparison_summary.csv"
)

PER_CLASS_CSV_PATH = (
    CSV_OUTPUT_DIR
    / "e3_tracker_per_class.csv"
)

CHART_OUTPUT_PATH = (
    CHART_OUTPUT_DIR
    / "e3_tracker_comparison.png"
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
# LINE GEOMETRY
# ==========================================================

def point_side(
    point,
    line_start,
    line_end,
):
    """
    Return side of a point relative to directed line.

    -1 = negative side
     0 = exactly on line
     1 = positive side
    """

    x, y = point

    x1, y1 = line_start
    x2, y2 = line_end

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def orientation(
    a,
    b,
    c,
):
    value = (
        (b[0] - a[0])
        * (c[1] - a[1])
        -
        (b[1] - a[1])
        * (c[0] - a[0])
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def on_segment(
    a,
    b,
    point,
):
    return (
        min(a[0], b[0])
        <= point[0]
        <= max(a[0], b[0])
        and
        min(a[1], b[1])
        <= point[1]
        <= max(a[1], b[1])
    )


def segments_intersect(
    a,
    b,
    c,
    d,
):
    """
    Check finite segment intersection.
    """

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)

    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if (
        o1 != o2
        and o3 != o4
    ):
        return True

    if (
        o1 == 0
        and on_segment(a, b, c)
    ):
        return True

    if (
        o2 == 0
        and on_segment(a, b, d)
    ):
        return True

    if (
        o3 == 0
        and on_segment(c, d, a)
    ):
        return True

    if (
        o4 == 0
        and on_segment(c, d, b)
    ):
        return True

    return False


# ==========================================================
# RUN ONE TRACKER
# ==========================================================

def run_tracker(
    tracker_label,
    tracker_type,
):

    print("\n")
    print("=" * 80)
    print(
        f"RUNNING: "
        f"YOLOv8s + {tracker_label}"
    )
    print("=" * 80)

    # ------------------------------------------------------
    # CREATE TRACKER
    # ------------------------------------------------------

    tracker = VehicleTracker(
        model_path=MODEL_PATH,
        tracker_type=tracker_type,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
    )

    # Thuận chưa truyền device vào model.track().
    # Move model trước để dùng GPU nếu có.
    tracker.model.to(
        DEVICE
    )

    # ------------------------------------------------------
    # OPEN VIDEO
    # ------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Khong mo duoc video: {VIDEO_PATH}"
        )

    video_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if video_fps <= 0:
        video_fps = 30.0

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
        f"Video: {VIDEO_PATH.name}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"Frames: "
        f"{total_frames}"
    )

    print(
        f"Video FPS: "
        f"{video_fps:.2f}"
    )

    print(
        f"Detector: YOLOv8s"
    )

    print(
        f"Tracker: {tracker_label}"
    )

    print(
        f"Confidence: "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"IoU: "
        f"{IOU_THRESHOLD}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}"
    )

    print(
        "Counting line:",
        f"{LINE_START} -> {LINE_END}",
    )

    print(
        "Ground Truth:",
        GROUND_TRUTH_TOTAL,
    )

    # ======================================================
    # COUNTING STATE
    # ======================================================

    counted_ids = set()

    class_counts = {
        class_name: 0
        for class_name
        in VEHICLE_CLASSES
    }

    track_last_side = {}

    track_last_point = {}

    counting_events = []


    # ======================================================
    # TRACKING QUALITY STATE
    # ======================================================

    unique_track_ids = set()

    track_frame_counts = Counter()

    track_first_frame = {}

    track_last_frame = {}

    track_class_history = defaultdict(
        list
    )

    total_track_observations = 0

    # Same heuristic idea as Thuận's tracker.py.
    #
    # IMPORTANT:
    # this is NOT standard ID Switch.
    last_observation = {}

    possible_fragmentation_events = []


    # ======================================================
    # PROCESS VIDEO
    # ======================================================

    frame_index = 0

    start_time = (
        time.perf_counter()
    )

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_index += 1

        # --------------------------------------------------
        # TRACK FRAME
        # --------------------------------------------------

        objects = tracker.track_frame(
            frame=frame,
            frame_number=frame_index,
            fps=video_fps,
        )


        # ==================================================
        # TRACKING QUALITY METRICS
        # ==================================================

        for obj in objects:

            total_track_observations += 1

            unique_track_ids.add(
                obj.track_id
            )

            track_frame_counts[
                obj.track_id
            ] += 1

            track_first_frame.setdefault(
                obj.track_id,
                frame_index,
            )

            track_last_frame[
                obj.track_id
            ] = frame_index

            track_class_history[
                obj.track_id
            ].append(
                obj.class_name
            )

            # ----------------------------------------------
            # HEURISTIC POSSIBLE FRAGMENTATION
            # ----------------------------------------------

            if (
                obj.track_id
                not in last_observation
            ):

                for (
                    old_id,
                    old_data,
                ) in last_observation.items():

                    (
                        old_frame,
                        old_x,
                        old_y,
                        old_class,
                    ) = old_data

                    gap = (
                        frame_index
                        - old_frame
                    )

                    distance = math.hypot(
                        obj.center_x
                        - old_x,
                        obj.center_y
                        - old_y,
                    )

                    if (
                        1 <= gap <= 30
                        and
                        distance <= 120
                        and
                        old_id != obj.track_id
                        and
                        old_class
                        == obj.class_name
                    ):

                        possible_fragmentation_events.append(
                            (
                                frame_index,
                                old_id,
                                obj.track_id,
                            )
                        )

                        break

            last_observation[
                obj.track_id
            ] = (
                frame_index,
                obj.center_x,
                obj.center_y,
                obj.class_name,
            )


        # ==================================================
        # COUNTING
        # ==================================================

        for obj in objects:

            if (
                obj.class_name
                not in VEHICLE_CLASSES
            ):
                continue

            track_id = (
                obj.track_id
            )

            center = (
                int(
                    round(
                        obj.center_x
                    )
                ),
                int(
                    round(
                        obj.center_y
                    )
                ),
            )

            current_side = (
                point_side(
                    center,
                    LINE_START,
                    LINE_END,
                )
            )

            # Do not overwrite last valid side
            # when center lies exactly on line.
            if current_side == 0:
                continue

            previous_side = (
                track_last_side.get(
                    track_id
                )
            )

            previous_point = (
                track_last_point.get(
                    track_id
                )
            )

            # ------------------------------------------------
            # Official rule:
            #
            # negative -> positive
            # LEFT -> RIGHT
            # ------------------------------------------------

            crossed_direction = (
                previous_side
                is not None
                and
                previous_side < 0
                and
                current_side > 0
            )

            crossed_finite_line = False

            if (
                crossed_direction
                and
                previous_point
                is not None
            ):

                crossed_finite_line = (
                    segments_intersect(
                        previous_point,
                        center,
                        LINE_START,
                        LINE_END,
                    )
                )

            # ------------------------------------------------
            # COUNT ONCE PER TRACK ID
            # ------------------------------------------------

            if (
                crossed_direction
                and
                crossed_finite_line
                and
                track_id
                not in counted_ids
            ):

                counted_ids.add(
                    track_id
                )

                class_counts[
                    obj.class_name
                ] += 1

                timestamp = (
                    (
                        frame_index - 1
                    )
                    / video_fps
                )

                counting_events.append(
                    {
                        "model":
                            "YOLOv8s",

                        "tracker":
                            tracker_label,

                        "timestamp":
                            round(
                                timestamp,
                                3,
                            ),

                        "frame_index":
                            frame_index,

                        "track_id":
                            track_id,

                        "class":
                            obj.class_name,

                        "line":
                            LINE_NAME,

                        "direction":
                            "LEFT_TO_RIGHT",

                        "confidence":
                            round(
                                obj.confidence,
                                4,
                            ),
                    }
                )

                print(
                    f"[{tracker_label}] "
                    f"COUNTED: "
                    f"{obj.class_name} "
                    f"ID={track_id} "
                    f"| t={timestamp:.2f}s "
                    f"| Total="
                    f"{sum(class_counts.values())}"
                )

            # ------------------------------------------------
            # UPDATE COUNTING HISTORY
            # ------------------------------------------------

            track_last_side[
                track_id
            ] = current_side

            track_last_point[
                track_id
            ] = center


    # ======================================================
    # END VIDEO
    # ======================================================

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    processing_fps = (
        frame_index
        / elapsed_time
        if elapsed_time > 0
        else 0
    )

    cap.release()


    # ======================================================
    # SAVE EVENT CSV
    # ======================================================

    tracker_slug = (
        tracker_label
        .lower()
        .replace("-", "")
    )

    event_csv_path = (
        CSV_OUTPUT_DIR
        / (
            f"e3_"
            f"{tracker_slug}_"
            f"eval02_events.csv"
        )
    )

    fieldnames = [
        "model",
        "tracker",
        "timestamp",
        "frame_index",
        "track_id",
        "class",
        "line",
        "direction",
        "confidence",
    ]

    with open(
        event_csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            counting_events
        )


    # ======================================================
    # COUNTING METRICS
    # ======================================================

    predicted_total = sum(
        class_counts.values()
    )

    total_absolute_error = abs(
        predicted_total
        - GROUND_TRUTH_TOTAL
    )

    total_count_accuracy = (
        1
        - (
            total_absolute_error
            / GROUND_TRUTH_TOTAL
        )
    )

    total_count_accuracy = max(
        0.0,
        total_count_accuracy,
    )


    # ======================================================
    # PER-CLASS METRICS
    # ======================================================

    class_absolute_errors = {}

    for class_name in (
        VEHICLE_CLASSES
    ):

        class_absolute_errors[
            class_name
        ] = abs(
            class_counts[
                class_name
            ]
            - GROUND_TRUTH[
                class_name
            ]
        )

    sum_class_absolute_error = sum(
        class_absolute_errors.values()
    )

    class_aware_accuracy = (
        1
        - (
            sum_class_absolute_error
            / GROUND_TRUTH_TOTAL
        )
    )

    class_aware_accuracy = max(
        0.0,
        class_aware_accuracy,
    )

    class_mae = (
        sum_class_absolute_error
        / len(
            VEHICLE_CLASSES
        )
    )


    # ======================================================
    # TRACK QUALITY METRICS
    # ======================================================

    track_lengths = list(
        track_frame_counts.values()
    )

    average_track_length = (
        sum(track_lengths)
        / len(track_lengths)
        if track_lengths
        else 0
    )

    gaps = []

    for track_id in (
        unique_track_ids
    ):

        first_frame = (
            track_first_frame[
                track_id
            ]
        )

        last_frame = (
            track_last_frame[
                track_id
            ]
        )

        observed_frames = (
            track_frame_counts[
                track_id
            ]
        )

        expected_frames = (
            last_frame
            - first_frame
            + 1
        )

        gaps.append(
            max(
                expected_frames
                - observed_frames,
                0,
            )
        )

    total_track_gaps = sum(
        gaps
    )

    average_track_gap = (
        total_track_gaps
        / len(gaps)
        if gaps
        else 0
    )

    class_transitions = sum(
        sum(
            class_a != class_b
            for class_a, class_b
            in zip(
                history,
                history[1:],
            )
        )
        for history
        in track_class_history.values()
    )

    unstable_class_tracks = sum(
        len(
            set(
                history
            )
        ) > 1
        for history
        in track_class_history.values()
        if history
    )


    # ======================================================
    # PRINT RESULT
    # ======================================================

    print("\n")
    print("-" * 80)
    print(
        f"KET QUA: "
        f"YOLOv8s + "
        f"{tracker_label}"
    )
    print("-" * 80)

    for class_name in (
        VEHICLE_CLASSES
    ):

        print(
            f"{class_name:12s}"
            f" | GT="
            f"{GROUND_TRUTH[class_name]:3d}"
            f" | Pred="
            f"{class_counts[class_name]:3d}"
            f" | AE="
            f"{class_absolute_errors[class_name]:3d}"
        )

    print("-" * 80)

    print(
        f"GT Total: "
        f"{GROUND_TRUTH_TOTAL}"
    )

    print(
        f"Pred Total: "
        f"{predicted_total}"
    )

    print(
        f"Total AE: "
        f"{total_absolute_error}"
    )

    print(
        "Total Counting Accuracy: "
        f"{total_count_accuracy * 100:.2f}%"
    )

    print(
        "Class-aware Counting Accuracy: "
        f"{class_aware_accuracy * 100:.2f}%"
    )

    print(
        f"Class MAE: "
        f"{class_mae:.2f}"
    )

    print(
        f"Unique Track IDs: "
        f"{len(unique_track_ids)}"
    )

    print(
        f"Average Track Length: "
        f"{average_track_length:.2f} frames"
    )

    print(
        f"Average Track Gap: "
        f"{average_track_gap:.2f} frames"
    )

    print(
        f"Class Transitions: "
        f"{class_transitions}"
    )

    print(
        f"Unstable Class Tracks: "
        f"{unstable_class_tracks}"
    )

    print(
        "Possible Fragmentation "
        "(heuristic): "
        f"{len(possible_fragmentation_events)}"
    )

    print(
        f"Processing FPS: "
        f"{processing_fps:.2f}"
    )

    print(
        f"Processing Time: "
        f"{elapsed_time:.2f}s"
    )

    print(
        f"Events CSV: "
        f"{event_csv_path}"
    )


    # ======================================================
    # RETURN
    # ======================================================

    return {
        "tracker":
            tracker_label,

        "counts":
            class_counts,

        "class_errors":
            class_absolute_errors,

        "predicted_total":
            predicted_total,

        "total_absolute_error":
            total_absolute_error,

        "total_count_accuracy":
            total_count_accuracy,

        "sum_class_absolute_error":
            sum_class_absolute_error,

        "class_aware_accuracy":
            class_aware_accuracy,

        "class_mae":
            class_mae,

        "unique_track_ids":
            len(
                unique_track_ids
            ),

        "average_track_length_frames":
            average_track_length,

        "total_track_gaps":
            total_track_gaps,

        "average_track_gap_frames":
            average_track_gap,

        "class_transitions":
            class_transitions,

        "unstable_class_tracks":
            unstable_class_tracks,

        "possible_fragmentation_events":
            len(
                possible_fragmentation_events
            ),

        "total_track_observations":
            total_track_observations,

        "processing_fps":
            processing_fps,

        "processing_time":
            elapsed_time,

        "events_csv":
            str(
                event_csv_path
            ),
    }


# ==========================================================
# SAVE SUMMARY
# ==========================================================

def save_summary(
    results,
):

    fieldnames = [
        "tracker",
        "model",
        "gt_total",
        "pred_total",
        "total_ae",
        "total_count_accuracy",
        "sum_class_ae",
        "class_aware_accuracy",
        "class_mae",
        "unique_track_ids",
        "average_track_length_frames",
        "total_track_gaps",
        "average_track_gap_frames",
        "class_transitions",
        "unstable_class_tracks",
        "possible_fragmentation_events",
        "processing_fps",
        "processing_time_s",
    ]

    with open(
        SUMMARY_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    "tracker":
                        result[
                            "tracker"
                        ],

                    "model":
                        "YOLOv8s",

                    "gt_total":
                        GROUND_TRUTH_TOTAL,

                    "pred_total":
                        result[
                            "predicted_total"
                        ],

                    "total_ae":
                        result[
                            "total_absolute_error"
                        ],

                    "total_count_accuracy":
                        round(
                            result[
                                "total_count_accuracy"
                            ]
                            * 100,
                            2,
                        ),

                    "sum_class_ae":
                        result[
                            "sum_class_absolute_error"
                        ],

                    "class_aware_accuracy":
                        round(
                            result[
                                "class_aware_accuracy"
                            ]
                            * 100,
                            2,
                        ),

                    "class_mae":
                        round(
                            result[
                                "class_mae"
                            ],
                            2,
                        ),

                    "unique_track_ids":
                        result[
                            "unique_track_ids"
                        ],

                    "average_track_length_frames":
                        round(
                            result[
                                "average_track_length_frames"
                            ],
                            2,
                        ),

                    "total_track_gaps":
                        result[
                            "total_track_gaps"
                        ],

                    "average_track_gap_frames":
                        round(
                            result[
                                "average_track_gap_frames"
                            ],
                            2,
                        ),

                    "class_transitions":
                        result[
                            "class_transitions"
                        ],

                    "unstable_class_tracks":
                        result[
                            "unstable_class_tracks"
                        ],

                    "possible_fragmentation_events":
                        result[
                            "possible_fragmentation_events"
                        ],

                    "processing_fps":
                        round(
                            result[
                                "processing_fps"
                            ],
                            2,
                        ),

                    "processing_time_s":
                        round(
                            result[
                                "processing_time"
                            ],
                            2,
                        ),
                }
            )


# ==========================================================
# SAVE PER CLASS
# ==========================================================

def save_per_class(
    results,
):

    fieldnames = [
        "tracker",
        "class",
        "ground_truth",
        "prediction",
        "absolute_error",
    ]

    with open(
        PER_CLASS_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            for class_name in (
                VEHICLE_CLASSES
            ):

                writer.writerow(
                    {
                        "tracker":
                            result[
                                "tracker"
                            ],

                        "class":
                            class_name,

                        "ground_truth":
                            GROUND_TRUTH[
                                class_name
                            ],

                        "prediction":
                            result[
                                "counts"
                            ][
                                class_name
                            ],

                        "absolute_error":
                            result[
                                "class_errors"
                            ][
                                class_name
                            ],
                    }
                )


# ==========================================================
# SAVE CHART
# ==========================================================

def save_chart(
    results,
):

    tracker_names = [
        result[
            "tracker"
        ]
        for result
        in results
    ]

    accuracies = [
        (
            result[
                "class_aware_accuracy"
            ]
            * 100
        )
        for result
        in results
    ]

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        tracker_names,
        accuracies,
    )

    plt.title(
        "E3 Tracker Comparison - "
        "Class-aware Counting Accuracy"
    )

    plt.xlabel(
        "Tracker"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.ylim(
        0,
        100,
    )

    plt.tight_layout()

    plt.savefig(
        CHART_OUTPUT_PATH,
        dpi=150,
    )

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay video: "
            f"{VIDEO_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay YOLOv8s: "
            f"{MODEL_PATH}"
        )

    print("\n")
    print("=" * 80)

    print(
        "EXPERIMENT 03 - "
        "TRACKER COMPARISON"
    )

    print("=" * 80)

    print(
        f"Detector: YOLOv8s"
    )

    print(
        "Trackers: "
        "ByteTrack vs BoT-SORT"
    )

    print(
        f"Evaluation video: "
        f"{VIDEO_PATH}"
    )

    print(
        "Counting line: "
        f"{LINE_START} -> "
        f"{LINE_END}"
    )

    print(
        f"Confidence: "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"IoU: "
        f"{IOU_THRESHOLD}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Ground Truth: "
        f"{GROUND_TRUTH_TOTAL}"
    )


    # ======================================================
    # RUN TRACKERS
    # ======================================================

    all_results = []

    for (
        tracker_label,
        tracker_type,
    ) in TRACKERS.items():

        result = run_tracker(
            tracker_label,
            tracker_type,
        )

        all_results.append(
            result
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ======================================================
    # SAVE RESULTS
    # ======================================================

    save_summary(
        all_results
    )

    save_per_class(
        all_results
    )

    save_chart(
        all_results
    )


    # ======================================================
    # SELECT BEST TRACKER
    # ======================================================

    # Priority:
    #
    # 1. Class-aware Counting Accuracy
    # 2. Total Counting Accuracy
    # 3. Fewer possible fragmentation events
    # 4. Higher processing FPS
    #
    # Fragmentation is only heuristic because
    # there is no tracking Ground Truth.

    best_result = max(
        all_results,
        key=lambda result: (
            result[
                "class_aware_accuracy"
            ],
            result[
                "total_count_accuracy"
            ],
            -result[
                "possible_fragmentation_events"
            ],
            result[
                "processing_fps"
            ],
        ),
    )


    # ======================================================
    # FINAL COMPARISON
    # ======================================================

    print("\n")
    print("=" * 120)

    print(
        "FINAL COMPARISON"
    )

    print("=" * 120)

    for result in (
        all_results
    ):

        print(
            f"{result['tracker']:10s}"
            f" | Total="
            f"{result['predicted_total']:3d}"
            f" | AE="
            f"{result['total_absolute_error']:3d}"
            f" | Total Acc="
            f"{result['total_count_accuracy'] * 100:6.2f}%"
            f" | Class-aware Acc="
            f"{result['class_aware_accuracy'] * 100:6.2f}%"
            f" | MAE="
            f"{result['class_mae']:5.2f}"
            f" | Unique IDs="
            f"{result['unique_track_ids']:4d}"
            f" | Avg Track="
            f"{result['average_track_length_frames']:6.2f}"
            f" | Avg Gap="
            f"{result['average_track_gap_frames']:6.2f}"
            f" | Frag.Heu.="
            f"{result['possible_fragmentation_events']:4d}"
            f" | FPS="
            f"{result['processing_fps']:6.2f}"
        )

    print("\n")
    print("=" * 80)

    print(
        "BEST TRACKER:",
        best_result[
            "tracker"
        ],
    )

    print(
        "Class-aware Counting Accuracy:",
        (
            f"{best_result['class_aware_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "Total Counting Accuracy:",
        (
            f"{best_result['total_count_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "Class MAE:",
        (
            f"{best_result['class_mae']:.2f}"
        ),
    )

    print(
        "Possible Fragmentation "
        "(heuristic):",
        best_result[
            "possible_fragmentation_events"
        ],
    )

    print(
        "Processing FPS:",
        (
            f"{best_result['processing_fps']:.2f}"
        ),
    )

    print("=" * 80)

    print(
        "\nNOTE:"
    )

    print(
        "possible_fragmentation_events "
        "chi la heuristic."
    )

    print(
        "Khong goi no la ID Switch, "
        "IDF1, MOTA hoac HOTA."
    )

    print(
        "Best Tracker duoc chon chu yeu "
        "dua tren downstream counting accuracy."
    )

    print("\nSaved:")

    print(
        SUMMARY_CSV_PATH
    )

    print(
        PER_CLASS_CSV_PATH
    )

    print(
        CHART_OUTPUT_PATH
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()