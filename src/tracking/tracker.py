from __future__ import annotations

import csv
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"


# ============================================================
# INPUT VIDEOS
# ============================================================
VIDEO_PATHS = [
    (
        "eval_03",
        PROJECT_ROOT
        / "data"
        / "videos"
        / "evaluation"
        / "eval_03.mp4",
    ),
    (
        "eval_02",
        PROJECT_ROOT
        / "data"
        / "videos"
        / "evaluation"
        / "eval_02.mp4",
    ),
    (
            "eval_01",
            PROJECT_ROOT
            / "data"
            / "videos"
            / "evaluation"
            / "eval_01_detrac_40131_40141.mp4",
    ),
]

OUTPUT_VIDEO_DIR = PROJECT_ROOT / "outputs" / "videos"
OUTPUT_CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
OUTPUT_RESULT_DIR = PROJECT_ROOT / "outputs" / "results"


# ============================================================
# VEHICLE CLASSES
# ============================================================
# COCO class IDs
# bicycle    = 1
# car        = 2
# motorcycle = 3
# bus        = 5
# truck      = 7
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class TrackObject:
    frame: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float


class VehicleTracker:
    SUPPORTED_TRACKERS = {
        "bytetrack": "bytetrack.yaml",
        "botsort": "botsort.yaml",
    }

    def __init__(
        self,
        model_path: str | Path,
        tracker_type: str = "bytetrack",
        conf: float = 0.4,
        iou: float = 0.7,
        imgsz: int = 640,
    ):
        tracker_type = tracker_type.lower()

        if tracker_type not in self.SUPPORTED_TRACKERS:
            raise ValueError(
                f"Tracker không hợp lệ: {tracker_type}. "
                f"Chỉ hỗ trợ: {list(self.SUPPORTED_TRACKERS.keys())}"
            )

        self.model_path = str(model_path)
        self.tracker_type = tracker_type
        self.tracker_config = self.SUPPORTED_TRACKERS[tracker_type]
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        print("\n" + "=" * 70)
        print(f"Loading YOLOv8n + {self.tracker_type}")
        print("=" * 70)

        self.model = YOLO(self.model_path)

    def track_frame(
        self,
        frame,
        frame_number: int,
        fps: float,
    ) -> list[TrackObject]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=VEHICLE_CLASSES,
            verbose=False,
        )

        result = results[0]
        objects: list[TrackObject] = []

        if result.boxes is None or len(result.boxes) == 0:
            return objects

        # Chỉ lưu object đã được tracker cấp ID.
        if result.boxes.id is None:
            return objects

        xyxy = result.boxes.xyxy.cpu().tolist()
        cls_ids = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        for bbox, cls_id, conf, track_id in zip(
            xyxy, cls_ids, confs, track_ids
        ):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            class_name = CLASS_NAMES.get(int(cls_id), str(cls_id))

            objects.append(
                TrackObject(
                    frame=frame_number,
                    timestamp=frame_number / fps,
                    track_id=int(track_id),
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    center_x=float(center_x),
                    center_y=float(center_y),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                )
            )

        return objects

    def process_video(
        self,
        input_video: str | Path,
        output_video: str | Path,
        output_csv: str | Path,
        show: bool = False,
    ) -> dict:
        input_video = Path(input_video)
        output_video = Path(output_video)
        output_csv = Path(output_csv)

        if not input_video.exists():
            raise FileNotFoundError(f"Không tìm thấy video: {input_video}")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Input      : {input_video}")
        print(f"Resolution : {width} x {height}")
        print(f"Frames     : {total_frames}")
        print(f"Video FPS  : {fps:.2f}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video), fourcc, fps, (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Không thể tạo output video: {output_video}")

        # --------------------------------------------------------
        # DETAIL CSV: 1 file / video / tracker
        # --------------------------------------------------------
        csv_file = open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8",
        )
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame",
                "timestamp",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "width",
                "height",
            ]
        )

        frame_count = 0
        total_track_objects = 0
        unique_track_ids: set[int] = set()
        class_counter = Counter()
        track_frame_counts = Counter()
        track_first_frame: dict[int, int] = {}
        track_last_frame: dict[int, int] = {}
        track_class_history = defaultdict(list)
        inference_times: list[float] = []

        # Heuristic fragmentation metric; NOT a standard ID Switch metric.
        last_observation = {}
        possible_fragmentation_events = []

        start_time = time.perf_counter()

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1

                inference_start = time.perf_counter()
                objects = self.track_frame(frame, frame_count, fps)
                inference_times.append(time.perf_counter() - inference_start)

                # ------------------------------------------------
                # TRACK QUALITY STATE
                # ------------------------------------------------
                for obj in objects:
                    if obj.track_id not in last_observation:
                        for old_id, (
                            old_frame,
                            old_x,
                            old_y,
                            old_cls,
                        ) in last_observation.items():
                            gap = obj.frame - old_frame
                            dist = math.hypot(
                                obj.center_x - old_x,
                                obj.center_y - old_y,
                            )
                            if (
                                1 <= gap <= 30
                                and dist <= 120
                                and old_id != obj.track_id
                            ):
                                possible_fragmentation_events.append(
                                    (obj.frame, old_id, obj.track_id)
                                )
                                break

                    last_observation[obj.track_id] = (
                        obj.frame,
                        obj.center_x,
                        obj.center_y,
                        obj.class_name,
                    )

                # ------------------------------------------------
                # SAVE TRACKING DATA
                # ------------------------------------------------
                for obj in objects:
                    total_track_objects += 1
                    unique_track_ids.add(obj.track_id)
                    class_counter[obj.class_name] += 1
                    track_frame_counts[obj.track_id] += 1
                    track_first_frame.setdefault(obj.track_id, obj.frame)
                    track_last_frame[obj.track_id] = obj.frame
                    track_class_history[obj.track_id].append(obj.class_name)

                    csv_writer.writerow(
                        [
                            obj.frame,
                            f"{obj.timestamp:.3f}",
                            obj.track_id,
                            obj.class_id,
                            obj.class_name,
                            f"{obj.confidence:.4f}",
                            f"{obj.x1:.2f}",
                            f"{obj.y1:.2f}",
                            f"{obj.x2:.2f}",
                            f"{obj.y2:.2f}",
                            f"{obj.center_x:.2f}",
                            f"{obj.center_y:.2f}",
                            f"{obj.width:.2f}",
                            f"{obj.height:.2f}",
                        ]
                    )

                # ------------------------------------------------
                # DRAW TRACKING RESULTS ONLY
                # ------------------------------------------------
                for obj in objects:
                    x1, y1 = int(obj.x1), int(obj.y1)
                    x2, y2 = int(obj.x2), int(obj.y2)
                    label = (
                        f"{obj.class_name} ID:{obj.track_id} "
                        f"{obj.confidence:.2f}"
                    )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame,
                        (int(obj.center_x), int(obj.center_y)),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                # Chỉ 3 dòng thông tin, không có nền đen lớn.
                def draw_text(text: str, x: int, y: int, scale: float = 0.65):
                    cv2.putText(
                        frame,
                        text,
                        (x + 2, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                draw_text(
                    f"YOLOv8n + {self.tracker_type}", 15, 30, 0.7
                )
                draw_text(
                    f"Frame: {frame_count}/{total_frames}", 15, 60, 0.6
                )
                draw_text(
                    f"Objects: {len(objects)}", 15, 90, 0.6
                )

                writer.write(frame)

                if show:
                    cv2.imshow("Traffic Tracking", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            csv_file.close()
            cv2.destroyAllWindows()

        total_time = time.perf_counter() - start_time
        processing_fps = frame_count / total_time if total_time > 0 else 0

        average_inference_time = (
            sum(inference_times) / len(inference_times)
            if inference_times
            else 0
        )
        inference_fps = (
            1 / average_inference_time
            if average_inference_time > 0
            else 0
        )

        track_lengths = list(track_frame_counts.values())
        average_track_length = (
            sum(track_lengths) / len(track_lengths)
            if track_lengths
            else 0
        )
        min_track_length = min(track_lengths) if track_lengths else 0
        max_track_length = max(track_lengths) if track_lengths else 0

        gaps = []
        for track_id in unique_track_ids:
            first = track_first_frame[track_id]
            last = track_last_frame[track_id]
            observed = track_frame_counts[track_id]
            expected = last - first + 1
            gaps.append(max(expected - observed, 0))

        total_track_gaps = sum(gaps)
        average_track_gap = (
            total_track_gaps / len(gaps) if gaps else 0
        )

        # Majority class per track, used only as a stable class summary.
        unique_ids_by_class = defaultdict(set)
        for track_id, history in track_class_history.items():
            if history:
                majority = Counter(history).most_common(1)[0][0]
                unique_ids_by_class[majority].add(track_id)

        class_transitions = sum(
            sum(a != b for a, b in zip(h, h[1:]))
            for h in track_class_history.values()
        )
        unstable_tracks = sum(
            len(set(h)) > 1 for h in track_class_history.values() if h
        )

        summary = {
            "tracker": self.tracker_type,
            "model": "YOLOv8n",
            "video": input_video.name,
            "frames": frame_count,
            "video_fps": round(fps, 2),
            "processing_fps": round(processing_fps, 2),
            "inference_fps": round(inference_fps, 2),
            "processing_time_seconds": round(total_time, 2),
            "total_track_observations": total_track_objects,
            "unique_track_ids": len(unique_track_ids),
            "average_track_length_frames": round(average_track_length, 2),
            "min_track_length_frames": min_track_length,
            "max_track_length_frames": max_track_length,
            "total_track_gaps": total_track_gaps,
            "average_track_gap_frames": round(average_track_gap, 2),
            "class_transitions": class_transitions,
            "unstable_class_tracks": unstable_tracks,
            "possible_fragmentation_events": len(possible_fragmentation_events),
            "bicycle_observations": class_counter["bicycle"],
            "car_observations": class_counter["car"],
            "motorcycle_observations": class_counter["motorcycle"],
            "bus_observations": class_counter["bus"],
            "truck_observations": class_counter["truck"],
            "unique_bicycle_ids": len(unique_ids_by_class["bicycle"]),
            "unique_car_ids": len(unique_ids_by_class["car"]),
            "unique_motorcycle_ids": len(unique_ids_by_class["motorcycle"]),
            "unique_bus_ids": len(unique_ids_by_class["bus"]),
            "unique_truck_ids": len(unique_ids_by_class["truck"]),
            "tracking_csv": str(output_csv),
            "tracking_video": str(output_video),
        }

        return summary


def clean_unnecessary_csvs():
    """Xóa các CSV cũ chỉ liên quan counting/fragmentation xuất bởi phiên bản trước."""
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    patterns = [
        "counting_events_*.csv",
        "counting_by_interval_*.csv",
        "possible_fragmentation_*.csv",
    ]
    for pattern in patterns:
        for path in OUTPUT_CSV_DIR.glob(pattern):
            try:
                path.unlink()
                print(f"[CLEAN] Removed: {path.name}")
            except OSError as exc:
                print(f"[WARN] Không thể xóa {path}: {exc}")



def save_comparison(summaries: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def print_comparison(summaries: list[dict]):
    print("\n" + "=" * 110)
    print("EXPERIMENT 03 - BYTETRACK vs BOT-SORT (TRACKING ONLY)")
    print("=" * 110)
    print(
        f"{'Video':<18}{'Tracker':<12}"
        f"{'Unique IDs':>12}{'Avg Track':>14}"
        f"{'Avg Gap':>12}{'Class Trans':>14}"
        f"{'Frag. Heu.':>12}{'Proc FPS':>12}"
    )
    print("-" * 110)

    for s in summaries:
        print(
            f"{s['video']:<18}{s['tracker']:<12}"
            f"{s['unique_track_ids']:>12}"
            f"{s['average_track_length_frames']:>14.2f}"
            f"{s['average_track_gap_frames']:>12.2f}"
            f"{s['class_transitions']:>14}"
            f"{s['possible_fragmentation_events']:>12}"
            f"{s['processing_fps']:>12.2f}"
        )

    print("=" * 110)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy YOLO model: {MODEL_PATH}")

    if not VIDEO_PATHS:
        raise ValueError("VIDEO_PATHS đang rỗng.")

    for video_name, video_path in VIDEO_PATHS:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy video [{video_name}]: {video_path}"
            )

    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Dọn CSV counting cũ để thư mục CSV chỉ còn dữ liệu cần cho Experiment 03.
    clean_unnecessary_csvs()

    summaries = []
    trackers = ["bytetrack", "botsort"]

    for video_name, video_path in VIDEO_PATHS:
        print("\n" + "#" * 90)
        print(f"VIDEO: {video_name}")
        print(f"INPUT: {video_path}")
        print("#" * 90)

        for tracker_name in trackers:
            tracker = VehicleTracker(
                model_path=MODEL_PATH,
                tracker_type=tracker_name,
                # Giữ giống nhau để so sánh công bằng.
                conf=0.4,
                iou=0.7,
                imgsz=640,
            )

            output_video = (
                OUTPUT_VIDEO_DIR
                / f"tracking_{video_name}_{tracker_name}.mp4"
            )
            output_csv = (
                OUTPUT_CSV_DIR
                / f"tracking_{video_name}_{tracker_name}.csv"
            )

            summary = tracker.process_video(
                input_video=video_path,
                output_video=output_video,
                output_csv=output_csv,
                show=False,
            )
            summary["video_group"] = video_name
            summaries.append(summary)

            print(
                f"[DONE] {video_name} - {tracker_name} | "
                f"Video: {output_video.name} | CSV: {output_csv.name}"
            )

    # Chỉ duy trì MỘT file tổng hợp cho toàn bộ Experiment 03.
    comparison_path = (
        OUTPUT_RESULT_DIR / "experiment_03_comparison_all_videos.csv"
    )
    save_comparison(summaries, comparison_path)
    print_comparison(summaries)

    print("\nCSV cần thiết sau khi chạy:")
    print("  - tracking_<video>_bytetrack.csv")
    print("  - tracking_<video>_botsort.csv")
    print("  - experiment_03_comparison_all_videos.csv")
    print("\nKhông xuất CSV riêng cho counting hoặc possible_fragmentation.")
    print("Lưu ý: possible_fragmentation_events chỉ là heuristic, không phải ID Switch chuẩn.")
    print("Không dùng IDF1/MOTA/HOTA nếu chưa có tracking ground truth.")


if __name__ == "__main__":
    main()
