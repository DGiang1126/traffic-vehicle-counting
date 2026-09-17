from pathlib import Path
from collections import defaultdict

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8n.pt"
)

CHART_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"
CHART_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHART_OUTPUT_PATH = (
    CHART_OUTPUT_DIR
    / "traffic_stats.png"
)


# ==========================================================
# VEHICLE CLASSES
# ==========================================================

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
}

# COCO class IDs
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
# DEVICE
# ==========================================================

if torch.cuda.is_available():
    DEVICE = 0

    print("CUDA available: True")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
else:
    DEVICE = "cpu"

    print("CUDA available: False")
    print("Dang chay bang CPU")


# ==========================================================
# MAIN
# ==========================================================

def main(
    video_path,
    model_path,
    line_y=400,
):

    # ======================================================
    # LOAD MODEL
    # ======================================================

    model = YOLO(
        str(model_path)
    )

    # ======================================================
    # OPEN VIDEO
    # ======================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Khong mo duoc video: {video_path}"
        )

    counted_ids = set()

    class_counts = defaultdict(int)

    track_history = {}

    # ======================================================
    # PROCESS VIDEO
    # ======================================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ==================================================
        # DETECTION + TRACKING
        # ==================================================

        results = model.track(
            frame,

            # giữ ID giữa các frame
            persist=True,

            # chỉ định rõ tracker baseline
            tracker="bytetrack.yaml",

            # chỉ giữ 5 class cần thiết
            classes=VEHICLE_CLASS_IDS,

            # confidence threshold
            conf=0.2,

            # 640 nhẹ hơn 960
            imgsz=640,

            # GPU 0 nếu có CUDA
            device=DEVICE,

            # FP16 nếu đang dùng GPU
            half=torch.cuda.is_available(),

            verbose=False,
        )[0]

        # ==================================================
        # COUNTING LINE
        # ==================================================

        cv2.line(
            frame,
            (0, line_y),
            (frame.shape[1], line_y),
            (0, 0, 255),
            2,
        )

        # ==================================================
        # VEHICLES
        # ==================================================

        if results.boxes.id is not None:

            for box, track_id in zip(
                results.boxes,
                results.boxes.id,
            ):

                # ------------------------------
                # CLASS
                # ------------------------------

                cls_id = int(
                    box.cls[0]
                )

                cls_name = (
                    model.names[cls_id]
                )

                if cls_name not in VEHICLE_CLASSES:
                    continue

                # ------------------------------
                # TRACK ID
                # ------------------------------

                track_id = int(
                    track_id
                )

                # ------------------------------
                # BOUNDING BOX
                # ------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0],
                )

                # Center Y
                cy = (
                    y1 + y2
                ) // 2

                # ==================================================
                # DRAW
                # ==================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{cls_name} #{track_id}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # ==================================================
                # COUNT
                # ==================================================

                last_cy = (
                    track_history.get(
                        track_id
                    )
                )

                if (
                    last_cy is not None
                    and last_cy < line_y <= cy
                    and track_id not in counted_ids
                ):

                    counted_ids.add(
                        track_id
                    )

                    class_counts[
                        cls_name
                    ] += 1

                    print(
                        f"COUNTED: "
                        f"{cls_name} "
                        f"ID={track_id} | "
                        f"Total="
                        f"{sum(class_counts.values())}"
                    )

                track_history[
                    track_id
                ] = cy

        # ==================================================
        # DISPLAY STATISTICS
        # ==================================================

        y_offset = 30

        cv2.putText(
            frame,
            f"Total: "
            f"{sum(class_counts.values())}",
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
                f"{cls_name}: "
                f"{class_counts[cls_name]}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # ==================================================
        # SHOW VIDEO
        # ==================================================

        cv2.imshow(
            "Traffic Counting - Press Q to quit",
            frame,
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break

    # ======================================================
    # CLEANUP
    # ======================================================

    cap.release()

    cv2.destroyAllWindows()

    # ======================================================
    # SAVE CHART
    # ======================================================

    if class_counts:

        labels = [
            "car",
            "motorcycle",
            "bus",
            "truck",
            "bicycle",
        ]

        values = [
            class_counts[label]
            for label in labels
        ]

        plt.figure()

        plt.bar(
            labels,
            values,
        )

        plt.title(
            "Traffic Vehicle Statistics"
        )

        plt.xlabel(
            "Vehicle Type"
        )

        plt.ylabel(
            "Count"
        )

        plt.tight_layout()

        plt.savefig(
            CHART_OUTPUT_PATH
        )

        plt.close()

        print(
            "\nDa luu bieu do tai:",
            CHART_OUTPUT_PATH,
        )

    print(
        "\nKet qua dem cuoi cung:"
    )

    for cls_name in [
        "car",
        "motorcycle",
        "bus",
        "truck",
        "bicycle",
    ]:

        print(
            f"{cls_name}: "
            f"{class_counts[cls_name]}"
        )

    print(
        "Total:",
        sum(
            class_counts.values()
        ),
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        line_y=700,
    )