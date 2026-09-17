from pathlib import Path
import csv
import time

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


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

MODEL_PATHS = {
    "YOLOv8n": PROJECT_ROOT / "models" / "yolov8n.pt",
    "YOLOv8s": PROJECT_ROOT / "models" / "yolov8s.pt",
    "YOLOv8m": PROJECT_ROOT / "models" / "yolov8m.pt",
}

CSV_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "csv"
CHART_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"

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
    / "e2_detector_comparison_summary.csv"
)

PER_CLASS_CSV_PATH = (
    CSV_OUTPUT_DIR
    / "e2_detector_per_class.csv"
)

CHART_OUTPUT_PATH = (
    CHART_OUTPUT_DIR
    / "e2_detector_comparison.png"
)


# ==========================================================
# EXPERIMENT CONFIG
# ==========================================================

# ----------------------------------------------------------
# Official counting line for eval_02
# Ground Truth by Kiet
# ----------------------------------------------------------

LINE_NAME = "line_1"

LINE_START = (66, 811)
LINE_END = (1744, 801)

# Looking from START -> END:
# only count crossing from LEFT side -> RIGHT side.
#
# In image coordinates (y increases downward), for this line:
# LEFT side  = negative side
# RIGHT side = positive side
#
# Therefore official direction:
# negative -> positive.

CONF_THRESHOLD = 0.2

IMAGE_SIZE = 640

TRACKER = "bytetrack.yaml"

SHOW_VIDEO = False


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

# COCO:
#
# bicycle    = 1
# car        = 2
# motorcycle = 3
# bus        = 5
# truck      = 7

VEHICLE_CLASS_IDS = [
    1,
    2,
    3,
    5,
    7,
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
# DEVICE
# ==========================================================

if torch.cuda.is_available():

    DEVICE = 0

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
    Determine which side of the directed line START -> END
    the point belongs to.

    Return:
        -1 : negative side
         0 : exactly on line
         1 : positive side
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
    """
    Orientation of three points.
    """

    value = (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def on_segment(
    a,
    b,
    p,
):
    """
    Check if p lies on finite segment a-b.
    """

    return (
        min(a[0], b[0])
        <= p[0]
        <= max(a[0], b[0])
        and
        min(a[1], b[1])
        <= p[1]
        <= max(a[1], b[1])
    )


def segments_intersect(
    a,
    b,
    c,
    d,
):
    """
    Check whether finite segment a-b
    intersects finite segment c-d.
    """

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    # Normal intersection
    if (
        o1 != o2
        and o3 != o4
    ):
        return True

    # Collinear edge cases
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
# RUN ONE MODEL
# ==========================================================

def run_model(
    model_name,
    model_path,
):

    print("\n")
    print("=" * 75)
    print(f"RUNNING: {model_name}")
    print("=" * 75)

    # ------------------------------------------------------
    # CHECK MODEL
    # ------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {model_path}"
        )

    # ------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------

    model = YOLO(
        str(model_path)
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

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    video_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    video_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Video resolution: "
        f"{video_width}x{video_height}"
    )

    print(
        "Counting line:",
        f"{LINE_START} -> {LINE_END}",
    )

    print(
        "Counting direction:",
        "LEFT -> RIGHT",
    )

    print(
        "Ground Truth:",
        GROUND_TRUTH_TOTAL,
    )

    # ------------------------------------------------------
    # CHECK LINE
    # ------------------------------------------------------

    for point in (
        LINE_START,
        LINE_END,
    ):

        if not (
            0 <= point[0] < video_width
            and
            0 <= point[1] < video_height
        ):

            raise ValueError(
                "Counting line nam ngoai "
                "kich thuoc video."
            )

    # ------------------------------------------------------
    # COUNTING STATE
    # ------------------------------------------------------

    # Track ID already counted
    counted_ids = set()

    class_counts = {
        cls_name: 0
        for cls_name
        in VEHICLE_CLASSES
    }

    # Last non-line center point
    track_last_point = {}

    # Last known side
    track_last_side = {}

    counting_events = []

    frame_index = 0

    start_time = (
        time.perf_counter()
    )

    # ======================================================
    # PROCESS VIDEO
    # ======================================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # --------------------------------------------------
        # DETECTION + TRACKING
        # --------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            classes=VEHICLE_CLASS_IDS,
            conf=CONF_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )[0]

        # --------------------------------------------------
        # DRAW COUNTING LINE
        # --------------------------------------------------

        if SHOW_VIDEO:

            cv2.line(
                frame,
                LINE_START,
                LINE_END,
                (0, 0, 255),
                3,
            )

            cv2.circle(
                frame,
                LINE_START,
                7,
                (0, 255, 255),
                -1,
            )

            cv2.circle(
                frame,
                LINE_END,
                7,
                (255, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                "START",
                (
                    LINE_START[0],
                    LINE_START[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "END",
                (
                    LINE_END[0] - 50,
                    LINE_END[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # --------------------------------------------------
        # PROCESS VEHICLES
        # --------------------------------------------------

        if results.boxes.id is not None:

            for (
                box,
                track_id_tensor,
            ) in zip(
                results.boxes,
                results.boxes.id,
            ):

                # ------------------------------------------
                # CLASS
                # ------------------------------------------

                cls_id = int(
                    box.cls[0]
                )

                cls_name = (
                    model.names[
                        cls_id
                    ]
                )

                if (
                    cls_name
                    not in VEHICLE_CLASSES
                ):
                    continue

                # ------------------------------------------
                # TRACK ID
                # ------------------------------------------

                track_id = int(
                    track_id_tensor
                )

                # ------------------------------------------
                # CONFIDENCE
                # ------------------------------------------

                confidence = float(
                    box.conf[0]
                )

                # ------------------------------------------
                # BBOX
                # ------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                cx = (
                    x1 + x2
                ) // 2

                cy = (
                    y1 + y2
                ) // 2

                center = (
                    cx,
                    cy,
                )

                # ------------------------------------------
                # DRAW
                # ------------------------------------------

                if SHOW_VIDEO:

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
                        (255, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"{cls_name} "
                            f"#{track_id} "
                            f"{confidence:.2f}"
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
                        (0, 255, 0),
                        1,
                    )

                # ------------------------------------------
                # SIDE OF COUNTING LINE
                # ------------------------------------------

                current_side = (
                    point_side(
                        center,
                        LINE_START,
                        LINE_END,
                    )
                )

                # Center exactly on line:
                # do not overwrite previous non-zero side.
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

                # ------------------------------------------
                # OFFICIAL CROSSING RULE
                #
                # START -> END
                # LEFT -> RIGHT
                #
                # For image coordinate system:
                # negative side -> positive side
                # ------------------------------------------

                crossed_direction = (
                    previous_side is not None
                    and
                    previous_side < 0
                    and
                    current_side > 0
                )

                # Must intersect FINITE line segment
                crossed_finite_segment = False

                if (
                    crossed_direction
                    and
                    previous_point
                    is not None
                ):

                    crossed_finite_segment = (
                        segments_intersect(
                            previous_point,
                            center,
                            LINE_START,
                            LINE_END,
                        )
                    )

                # ------------------------------------------
                # COUNT
                # ------------------------------------------

                if (
                    crossed_direction
                    and
                    crossed_finite_segment
                    and
                    track_id
                    not in counted_ids
                ):

                    counted_ids.add(
                        track_id
                    )

                    class_counts[
                        cls_name
                    ] += 1

                    timestamp = (
                        (
                            frame_index - 1
                        )
                        / video_fps
                        if video_fps > 0
                        else 0
                    )

                    counting_events.append(
                        {
                            "model": (
                                model_name
                            ),
                            "timestamp": round(
                                timestamp,
                                3,
                            ),
                            "frame_index": (
                                frame_index
                            ),
                            "track_id": (
                                track_id
                            ),
                            "class": (
                                cls_name
                            ),
                            "line": (
                                LINE_NAME
                            ),
                            "direction": (
                                "LEFT_TO_RIGHT"
                            ),
                            "confidence": round(
                                confidence,
                                4,
                            ),
                        }
                    )

                    print(
                        f"[{model_name}] "
                        f"COUNTED: "
                        f"{cls_name} "
                        f"ID={track_id} "
                        f"| t={timestamp:.2f}s "
                        f"| Total="
                        f"{sum(class_counts.values())}"
                    )

                # ------------------------------------------
                # UPDATE TRACK STATE
                # ------------------------------------------

                track_last_side[
                    track_id
                ] = current_side

                track_last_point[
                    track_id
                ] = center

        # --------------------------------------------------
        # SHOW VIDEO
        # --------------------------------------------------

        if SHOW_VIDEO:

            y_offset = 30

            cv2.putText(
                frame,
                (
                    f"{model_name} | "
                    f"Total: "
                    f"{sum(class_counts.values())}"
                ),
                (
                    10,
                    y_offset,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            for cls_name in (
                VEHICLE_CLASSES
            ):

                y_offset += 25

                cv2.putText(
                    frame,
                    (
                        f"{cls_name}: "
                        f"{class_counts[cls_name]}"
                    ),
                    (
                        10,
                        y_offset,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow(
                f"E2 - {model_name}",
                frame,
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
                break

    # ======================================================
    # END PROCESSING
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

    cv2.destroyAllWindows()

    # ======================================================
    # SAVE EVENT CSV
    # ======================================================

    event_csv_path = (
        CSV_OUTPUT_DIR
        / (
            f"e2_"
            f"{model_name.lower()}_"
            f"eval02_events.csv"
        )
    )

    fieldnames = [
        "model",
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
    # METRICS
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

    # ------------------------------------------------------
    # PER-CLASS ERROR
    # ------------------------------------------------------

    class_absolute_errors = {}

    for cls_name in (
        VEHICLE_CLASSES
    ):

        class_absolute_errors[
            cls_name
        ] = abs(
            class_counts[
                cls_name
            ]
            - GROUND_TRUTH[
                cls_name
            ]
        )

    sum_class_absolute_error = sum(
        class_absolute_errors.values()
    )

    # Custom downstream metric:
    # penalizes class-level count mismatch.
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
    # PRINT RESULT
    # ======================================================

    print("\n")
    print("-" * 75)
    print(
        f"KET QUA: "
        f"{model_name}"
    )
    print("-" * 75)

    for cls_name in (
        VEHICLE_CLASSES
    ):

        print(
            f"{cls_name:12s}"
            f" | GT="
            f"{GROUND_TRUTH[cls_name]:3d}"
            f" | Pred="
            f"{class_counts[cls_name]:3d}"
            f" | AE="
            f"{class_absolute_errors[cls_name]:3d}"
        )

    print("-" * 75)

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
        f"Processing FPS: "
        f"{processing_fps:.2f}"
    )

    print(
        f"Processing time: "
        f"{elapsed_time:.2f}s"
    )

    print(
        f"Events CSV: "
        f"{event_csv_path}"
    )

    return {
        "model": (
            model_name
        ),
        "counts": (
            class_counts
        ),
        "class_errors": (
            class_absolute_errors
        ),
        "predicted_total": (
            predicted_total
        ),
        "total_absolute_error": (
            total_absolute_error
        ),
        "total_count_accuracy": (
            total_count_accuracy
        ),
        "sum_class_absolute_error": (
            sum_class_absolute_error
        ),
        "class_aware_accuracy": (
            class_aware_accuracy
        ),
        "class_mae": (
            class_mae
        ),
        "processing_fps": (
            processing_fps
        ),
        "processing_time": (
            elapsed_time
        ),
    }


# ==========================================================
# SAVE SUMMARY
# ==========================================================

def save_summary(
    results,
):

    fieldnames = [
        "model",
        "gt_total",
        "pred_total",
        "total_ae",
        "total_count_accuracy",
        "sum_class_ae",
        "class_aware_accuracy",
        "class_mae",
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
                    "model":
                        result[
                            "model"
                        ],

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
# SAVE PER-CLASS
# ==========================================================

def save_per_class(
    results,
):

    fieldnames = [
        "model",
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

            for cls_name in (
                VEHICLE_CLASSES
            ):

                writer.writerow(
                    {
                        "model":
                            result[
                                "model"
                            ],

                        "class":
                            cls_name,

                        "ground_truth":
                            GROUND_TRUTH[
                                cls_name
                            ],

                        "prediction":
                            result[
                                "counts"
                            ][
                                cls_name
                            ],

                        "absolute_error":
                            result[
                                "class_errors"
                            ][
                                cls_name
                            ],
                    }
                )


# ==========================================================
# SAVE CHART
# ==========================================================

def save_chart(
    results,
):

    model_names = [
        result["model"]
        for result
        in results
    ]

    class_accuracies = [
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
        figsize=(8, 5)
    )

    plt.bar(
        model_names,
        class_accuracies,
    )

    plt.title(
        "E2 Detector Comparison - "
        "Class-aware Counting Accuracy"
    )

    plt.xlabel(
        "Detector"
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

    print("\n")
    print("=" * 75)
    print(
        "EXPERIMENT 2 - "
        "DETECTOR COMPARISON"
    )
    print("=" * 75)

    print(
        f"Evaluation video: "
        f"{VIDEO_PATH}"
    )

    print(
        f"Counting line: "
        f"{LINE_START} -> "
        f"{LINE_END}"
    )

    print(
        "Counting direction: "
        "LEFT -> RIGHT"
    )

    print(
        f"Tracker: "
        f"{TRACKER}"
    )

    print(
        f"Confidence: "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}"
    )

    print(
        f"GT car: "
        f"{GROUND_TRUTH['car']}"
    )

    print(
        f"GT motorcycle: "
        f"{GROUND_TRUTH['motorcycle']}"
    )

    print(
        f"GT bus: "
        f"{GROUND_TRUTH['bus']}"
    )

    print(
        f"GT truck: "
        f"{GROUND_TRUTH['truck']}"
    )

    print(
        f"GT bicycle: "
        f"{GROUND_TRUTH['bicycle']}"
    )

    print(
        f"Ground Truth Total: "
        f"{GROUND_TRUTH_TOTAL}"
    )

    # ------------------------------------------------------
    # RUN ALL MODELS
    # ------------------------------------------------------

    all_results = []

    for (
        model_name,
        model_path,
    ) in MODEL_PATHS.items():

        result = run_model(
            model_name=model_name,
            model_path=model_path,
        )

        all_results.append(
            result
        )

        if (
            torch.cuda.is_available()
        ):

            torch.cuda.empty_cache()

    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

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
    # SELECT BEST DETECTOR
    # ======================================================

    # Priority:
    #
    # 1. highest class-aware counting accuracy
    # 2. highest total counting accuracy
    # 3. higher processing FPS

    best_result = max(
        all_results,
        key=lambda x: (
            x[
                "class_aware_accuracy"
            ],
            x[
                "total_count_accuracy"
            ],
            x[
                "processing_fps"
            ],
        ),
    )

    # ======================================================
    # FINAL COMPARISON
    # ======================================================

    print("\n")
    print("=" * 75)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 75)

    for result in (
        all_results
    ):

        print(
            f"{result['model']:8s}"
            f" | Total="
            f"{result['predicted_total']:3d}"
            f" | Total AE="
            f"{result['total_absolute_error']:3d}"
            f" | Total Acc="
            f"{result['total_count_accuracy'] * 100:6.2f}%"
            f" | Class-aware Acc="
            f"{result['class_aware_accuracy'] * 100:6.2f}%"
            f" | Class MAE="
            f"{result['class_mae']:5.2f}"
            f" | FPS="
            f"{result['processing_fps']:6.2f}"
        )

    print("\n")
    print("=" * 75)

    print(
        "BEST DETECTOR:",
        best_result[
            "model"
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
        "Processing FPS:",
        (
            f"{best_result['processing_fps']:.2f}"
        ),
    )

    print("=" * 75)

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