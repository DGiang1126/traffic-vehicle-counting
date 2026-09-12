#Code starter
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ==========================================================
# PROJECT PATHS
# ==========================================================

# traffic-vehicle-counting/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "traffic.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"

CHART_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"
CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHART_OUTPUT_PATH = CHART_OUTPUT_DIR / "traffic_stats.png"


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


def main(
    video_path,
    model_path,
    line_y=400,
):

    # Load YOLO model
    model = YOLO(str(model_path))

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Khong mo duoc video: {video_path}"
        )

    # Các ID đã được đếm
    counted_ids = set()

    # Số lượng theo class
    class_counts = defaultdict(int)

    # {track_id: last_cy}
    track_history = {}

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ==================================================
        # DETECTION + TRACKING
        # ==================================================

        results = model.track(
            frame,
            persist=True,
            conf=0.2,
            imgsz=960,
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

        if results.boxes.id is not None:

            for box, track_id in zip(
                results.boxes,
                results.boxes.id,
            ):

                # Class
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name not in VEHICLE_CLASSES:
                    continue

                # Track ID
                track_id = int(track_id)

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0],
                )

                # Center Y
                cy = (y1 + y2) // 2

                # ==================================================
                # DRAW BOUNDING BOX
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
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # ==================================================
                # COUNT VEHICLE
                # ==================================================

                last_cy = track_history.get(track_id)

                if (
                    last_cy is not None
                    and last_cy < line_y <= cy
                    and track_id not in counted_ids
                ):
                    counted_ids.add(track_id)

                    class_counts[cls_name] += 1

                    print(
                        f"COUNTED: {cls_name} "
                        f"ID={track_id} | "
                        f"Total={sum(class_counts.values())}"
                    )

                track_history[track_id] = cy

        # ==================================================
        # DISPLAY STATISTICS
        # ==================================================

        y_offset = 30

        cv2.putText(
            frame,
            f"Total: {sum(class_counts.values())}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        for cls_name, count in class_counts.items():

            y_offset += 25

            cv2.putText(
                frame,
                f"{cls_name}: {count}",
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
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

        plt.figure()

        plt.bar(
            class_counts.keys(),
            class_counts.values(),
        )

        plt.title("Traffic Vehicle Statistics")
        plt.xlabel("Vehicle Type")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(CHART_OUTPUT_PATH)

        plt.close()

        print(
            f"\nDa luu bieu do tai: "
            f"{CHART_OUTPUT_PATH}"
        )

    print(
        "\nKet qua dem cuoi cung:",
        dict(class_counts),
    )


if __name__ == "__main__":

    main(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        line_y=400,
    )