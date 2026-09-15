"""Generic counting-line setup tool for any traffic video.

Run directly to choose a video, configure START/END lines, and save a reusable
line config. The manual Ground Truth tool imports the same functions.
"""

from pathlib import Path
import argparse
import json
import sys

try:
    import cv2
except ImportError:
    print("Missing opencv-python. Run: python -m pip install -r requirements.txt")
    raise

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = PROJECT_ROOT / "data" / "videos"
CONFIG_DIR = PROJECT_ROOT / "data" / "ground_truth" / "line_configs"
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 820
WINDOW_NAME = "Counting Line Setup"
COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]

def discover_videos(root=VIDEO_ROOT):
    videos = []
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                videos.append(path)
    return sorted(videos, key=lambda p: str(p).lower())


def choose_video_path(requested=None):
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            candidate = PROJECT_ROOT / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return path.resolve()

    videos = discover_videos()
    if not videos:
        raise FileNotFoundError(f"No videos found under: {VIDEO_ROOT}")
    print("\n=== CHON VIDEO ===")
    for i, path in enumerate(videos, 1):
        print(f"  {i}. {path.relative_to(PROJECT_ROOT)}")
    while True:
        raw = input("Nhap so video: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            return videos[int(raw) - 1].resolve()
        print("Lua chon khong hop le.")

def read_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return {
        "fps": fps,
        "frame_count": frames,
        "duration": frames / fps,
        "width": width,
        "height": height,
    }


def load_reference_frame(video_path):
    info = read_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(info["frame_count"] // 2, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from: {video_path}")
    return frame, info

def config_path_for(video_path):
    return CONFIG_DIR / f"{video_path.stem}_lines.json"


def normalized_point(point, width, height):
    return [round(point[0] / width, 6), round(point[1] / height, 6)]


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_saved_line_config(video_path):
    path = config_path_for(video_path)
    if path.exists():
        data = load_json(path)
        if data and data.get("lines"):
            return data, path
    return None, path

def ask_positive_int(prompt, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return int(default)
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Can nhap so nguyen > 0.")


def default_line(index, count, width, height):
    y = int(round(height * (index + 1) / (count + 1)))
    margin = max(10, int(width * 0.15))
    return {
        "name": f"line_{index + 1}",
        "start": [margin, y],
        "end": [width - margin, y],
    }


def parse_line_coordinates(raw, width, height):
    values = raw.replace(",", " ").split()
    if len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(int, values)
    except ValueError:
        return None
    if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
        return None
    if x1 == x2 and y1 == y2:
        return None
    return [x1, y1], [x2, y2]

def build_initial_lines(info, saved=None):
    existing = saved.get("lines", []) if saved else []
    default_count = len(existing) if existing else 1
    count = ask_positive_int("So counting line", default_count)
    lines = []
    print("\nNhap toa do theo dang: x1 y1 x2 y2")
    print("Enter = dung toa do cu/default roi chinh bang chuot o buoc sau.")
    for idx in range(count):
        if idx < len(existing):
            base = {
                "name": f"line_{idx + 1}",
                "start": list(map(int, existing[idx]["start"])),
                "end": list(map(int, existing[idx]["end"])),
            }
        else:
            base = default_line(idx, count, info["width"], info["height"])
        while True:
            raw = input(
                f"line_{idx + 1} START/END "
                f"[{base['start'][0]} {base['start'][1]} {base['end'][0]} {base['end'][1]}]: "
            ).strip()
            if raw == "":
                lines.append(base)
                break
            parsed = parse_line_coordinates(raw, info["width"], info["height"])
            if parsed:
                start, end = parsed
                lines.append({"name": f"line_{idx + 1}", "start": start, "end": end})
                break
            print("Toa do khong hop le hoac START trung END.")
    return lines

def draw_lines(frame, lines):
    canvas = frame.copy()
    for idx, line in enumerate(lines):
        color = COLORS[idx % len(COLORS)]
        p1 = tuple(map(int, line["start"]))
        p2 = tuple(map(int, line["end"]))
        cv2.line(canvas, p1, p2, color, 5, cv2.LINE_AA)
        cv2.circle(canvas, p1, 12, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (p2[0] - 11, p2[1] - 11),
                      (p2[0] + 11, p2[1] + 11), color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (p2[0] - 11, p2[1] - 11),
                      (p2[0] + 11, p2[1] + 11), (255, 255, 255), 2, cv2.LINE_AA)
        for label, point in ((f"{line['name']} START {p1}", p1), (f"END {p2}", p2)):
            cv2.putText(canvas, label, (point[0] + 12, max(25, point[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (point[0] + 12, max(25, point[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return canvas


class GenericLineEditor:
    def __init__(self, frame, lines):
        self.frame = frame
        self.lines = lines
        self.dragging = None
        self.scale = 1.0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

    def display_scale(self):
        h, w = self.frame.shape[:2]
        return min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)

    def to_original(self, x, y):
        return int(round(x / self.scale)), int(round(y / self.scale))

    def find_endpoint(self, x, y):
        ox, oy = self.to_original(x, y)
        threshold = max(18 / self.scale, 12)
        best = None
        best_d2 = threshold * threshold
        for li, line in enumerate(self.lines):
            for endpoint in ("start", "end"):
                px, py = line[endpoint]
                d2 = (ox - px) ** 2 + (oy - py) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (li, endpoint)
        return best

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = self.find_endpoint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            li, endpoint = self.dragging
            ox, oy = self.to_original(x, y)
            h, w = self.frame.shape[:2]
            ox = max(0, min(w - 1, ox))
            oy = max(0, min(h - 1, oy))
            self.lines[li][endpoint] = [ox, oy]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def render(self):
        canvas = draw_lines(self.frame, self.lines)
        self.scale = self.display_scale()
        if self.scale < 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(canvas, "Circle=START | Square=END | Drag endpoints",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(canvas, "ENTER=confirm | Q/Esc=cancel",
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        return canvas

    def run(self):
        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                cv2.destroyWindow(WINDOW_NAME)
                return self.lines
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(WINDOW_NAME)
                return None

def save_line_config(video_path, lines, info):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stored_video_path = str(video_path.relative_to(PROJECT_ROOT))
    except ValueError:
        stored_video_path = str(video_path)
    payload = {
        "video": video_path.name,
        "video_path": stored_video_path,
        "resolution": [info["width"], info["height"]],
        "confirmed": True,
        "lines": [],
    }
    for line in lines:
        payload["lines"].append({
            "name": line["name"],
            "start": list(map(int, line["start"])),
            "end": list(map(int, line["end"])),
            "start_normalized": normalized_point(line["start"], info["width"], info["height"]),
            "end_normalized": normalized_point(line["end"], info["width"], info["height"]),
        })
    path = config_path_for(video_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, path


def setup_lines(video_path, force_edit=False):
    frame, info = load_reference_frame(video_path)
    saved, source = load_saved_line_config(video_path)
    if saved and not force_edit:
        print(f"\nDa co line config: {source}")
        for line in saved["lines"]:
            print(f"  {line['name']}: START {line['start']} -> END {line['end']}")
        choice = input("Enter = dung lai | E = sua line: ").strip().lower()
        if choice != "e":
            return saved, source
    lines = build_initial_lines(info, saved)
    edited = GenericLineEditor(frame, lines).run()
    if edited is None:
        raise KeyboardInterrupt
    return save_line_config(video_path, edited, info)

def parse_args():
    parser = argparse.ArgumentParser(description="Set START/END counting lines for any video.")
    parser.add_argument("--video", help="Video path. Omit to choose interactively.")
    parser.add_argument("--edit", action="store_true", help="Edit even when a saved config exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = choose_video_path(args.video)
    config, path = setup_lines(video_path, force_edit=args.edit)
    print("\nLine setup complete")
    print(f"Video: {video_path}")
    print(f"Saved config: {path}")
    for line in config["lines"]:
        print(f"  {line['name']}: START {line['start']} -> END {line['end']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nCancelled.")
        sys.exit(130)
