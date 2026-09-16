"""Main multi-line vehicle-counting pipeline and project entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from src.counting.counter import LineDefinition, MultiLineCounter, normal_direction
from src.statistics.counting_statistics import save_statistics
from src.visualization.counting_renderer import render_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "traffic.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "e4_multi_line.json"


@dataclass(frozen=True)
class TrackedVehicle:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "vehicle_counting"


def _code_version() -> str:
    """Return a short fingerprint of the files that affect counting results."""

    digest = hashlib.sha256()
    relative_paths = [
        "src/traffic_counting.py",
        "src/counting/counter.py",
        "src/statistics/counting_statistics.py",
        "src/visualization/counting_renderer.py",
    ]
    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _unique_run_name(output_dir: Path, requested_name: str) -> str:
    """Append _02, _03... when any artifact with the same name exists."""

    base_name = _safe_name(requested_name)

    def already_exists(candidate: str) -> bool:
        return any(
            path.exists()
            for path in (
                output_dir / "videos" / f"{candidate}.mp4",
                output_dir / "csv" / f"{candidate}_events.csv",
                output_dir / "csv" / candidate,
                output_dir / "charts" / candidate,
            )
        )

    if not already_exists(base_name):
        return base_name
    suffix = 2
    while already_exists(f"{base_name}_{suffix:02d}"):
        suffix += 1
    return f"{base_name}_{suffix:02d}"


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pixel_point(point, width: int, height: int) -> tuple[int, int]:
    x, y = float(point[0]), float(point[1])
    if 0 <= x <= 1 and 0 <= y <= 1:
        return (round(x * width), round(y * height))
    return (round(x), round(y))


def build_lines(config: dict, width: int, height: int) -> list[LineDefinition]:
    definitions = []
    for item in config.get("counting_lines", []):
        start = _pixel_point(item["start"], width, height)
        end = _pixel_point(item["end"], width, height)
        raw_vector = item.get("direction_vector")
        if raw_vector is None:
            vector = normal_direction(start, end, bool(item.get("reverse_direction", False)))
        else:
            vx, vy = float(raw_vector[0]), float(raw_vector[1])
            length = math.hypot(vx, vy)
            if length == 0:
                raise ValueError(f"direction_vector của {item.get('name')} không được bằng 0")
            vector = (vx / length, vy / length)

        definitions.append(
            LineDefinition(
                name=str(item["name"]),
                start=start,
                end=end,
                direction_vector=vector,
                in_label=str(item.get("in_label", "IN")),
                out_label=str(item.get("out_label", "OUT")),
                negative_to_positive=str(item.get("negative_to_positive", "negative_to_positive")),
                positive_to_negative=str(item.get("positive_to_negative", "positive_to_negative")),
            )
        )
    return definitions


def _tracked_vehicles(result, model_names, vehicle_classes: set[str]) -> list[TrackedVehicle]:
    boxes = result.boxes
    if boxes.id is None:
        return []
    vehicles = []
    for box, raw_track_id in zip(boxes, boxes.id):
        class_name = model_names[int(box.cls[0])]
        if class_name not in vehicle_classes:
            continue
        vehicles.append(
            TrackedVehicle(
                track_id=int(raw_track_id),
                class_name=class_name,
                confidence=float(box.conf[0]),
                bbox=tuple(map(int, box.xyxy[0].tolist())),
            )
        )
    return vehicles


def _convert_to_h264(video_path: Path) -> tuple[bool, str | None]:
    """Make the MP4 browser-friendly when FFmpeg with libx264 is available."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "Không tìm thấy FFmpeg; video vẫn được giữ ở codec mp4v."
    converted_path = video_path.with_name(f"{video_path.stem}_h264.mp4")
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(converted_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode == 0 and converted_path.exists() and converted_path.stat().st_size > 0:
        converted_path.replace(video_path)
        return True, None
    converted_path.unlink(missing_ok=True)
    last_error = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "lỗi không xác định"
    return False, f"FFmpeg không chuyển được H.264 ({last_error}); giữ video mp4v."


def run(
    video_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = PROJECT_ROOT / "outputs",
    run_name: str | None = None,
    show: bool = False,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """Process one video and return paths to every generated artifact."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Thiếu Ultralytics. Chạy: pip install -r requirements-streamlit.txt") from exc

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    config = load_config(config_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    requested_run_name = _safe_name(
        run_name or f"{video_path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_name = _unique_run_name(output_dir, requested_run_name)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_version = _code_version()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lines = build_lines(config, width, height)
    if not lines:
        cap.release()
        raise ValueError("Config phải có ít nhất một phần tử trong counting_lines")
    counter = MultiLineCounter(lines)

    video_dir = output_dir / "videos"
    csv_dir = output_dir / "csv"
    chart_dir = output_dir / "charts" / run_name
    summary_dir = csv_dir / run_name
    for directory in (video_dir, csv_dir, chart_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_video = video_dir / f"{run_name}.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được video đầu ra: {output_video}")

    model_path = config.get("model", "yolov8n.pt")
    if not Path(str(model_path)).is_absolute() and (PROJECT_ROOT / str(model_path)).exists():
        model_path = str(PROJECT_ROOT / str(model_path))
    model = YOLO(str(model_path))
    vehicle_classes = set(config.get("vehicle_classes", ["car", "motorcycle", "bus", "truck", "bicycle"]))
    counts: Counter = Counter()
    events = []
    processed_frames = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            processed_frames += 1
            result = model.track(
                frame,
                persist=True,
                tracker=str(config.get("tracker", "bytetrack.yaml")),
                conf=float(config.get("confidence", 0.25)),
                imgsz=int(config.get("image_size", 960)),
                verbose=False,
            )[0]
            vehicles = _tracked_vehicles(result, model.names, vehicle_classes)
            timestamp = (processed_frames - 1) / fps
            for vehicle in vehicles:
                new_events = counter.update(
                    track_id=vehicle.track_id,
                    center=vehicle.center,
                    class_name=vehicle.class_name,
                    confidence=vehicle.confidence,
                    timestamp=timestamp,
                    frame_index=processed_frames - 1,
                )
                for event in new_events:
                    events.append(event)
                    counts[(event.line, event.class_name, event.direction)] += 1

            render_frame(frame, lines, vehicles, counts, len(events))
            writer.write(frame)
            if show:
                cv2.imshow("Multi-line Vehicle Counting (Q: quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if progress_callback and total_frames > 0:
                progress_callback(min(processed_frames / total_frames, 1.0))
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    browser_compatible, video_warning = _convert_to_h264(output_video)

    events_csv = csv_dir / f"{run_name}_events.csv"
    statistics = save_statistics(events, events_csv, summary_dir, chart_dir)
    config_snapshot = summary_dir / "config_used.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_version": code_version,
        "requested_run_name": requested_run_name,
        "run_name": run_name,
        "input_video": str(video_path),
        "config": str(config_path),
        "processed_frames": processed_frames,
        "fps": fps,
        "number_of_lines": len(lines),
        "line_names": [line.name for line in lines],
        "total_events": len(events),
        "config_snapshot": str(config_snapshot),
        "output_video": str(output_video),
        "browser_compatible_video": browser_compatible,
        "video_warning": video_warning,
        **statistics,
    }
    metrics_path = summary_dir / "run_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["metrics_json"] = str(metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-line vehicle counting")
    parser.add_argument("--video", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON config")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--run-name", default=None, help="Tên đầu ra; mặc định có timestamp")
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ OpenCV khi chạy")
    args = parser.parse_args()
    result = run(args.video, args.config, args.output_dir, args.run_name, args.show)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
