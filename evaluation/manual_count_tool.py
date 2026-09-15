from pathlib import Path
import argparse
import csv
import json
import math
import sys

try:
    import cv2
except ImportError:
    print("Missing opencv-python. Run: python -m pip install -r requirements.txt")
    raise

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"

CLASSES = ["motorcycle", "car", "bus", "truck", "bicycle"]
DEFAULT_INTERVAL = 10.0
WINDOW_NAME = "Manual Ground Truth Counter"
MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 820

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.line_editor import choose_video_path, setup_lines

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps,
        "width": width,
        "height": height,
    }


def make_segments(duration, interval):
    result = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + interval, duration)
        result.append((round(start, 3), round(end, 3)))
        start = end
    return result

def segment_key(start, end):
    return f"{start:.3f}-{end:.3f}"


def load_progress(path, video_id, video_file, interval):
    if not path.exists():
        return {
            "video_id": video_id,
            "video": video_file,
            "interval_sec": interval,
            "counts": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "video_id": video_id,
            "video": video_file,
            "interval_sec": interval,
            "counts": {},
        }
    data.setdefault("counts", {})
    return data


def line_signature(line):
    return {
        "start": list(map(int, line["start"])),
        "end": list(map(int, line["end"])),
    }


def sync_line_geometry(progress, lines):
    saved = progress.setdefault("line_geometry", {})
    changed = []
    counts = progress.setdefault("counts", {})
    current_names = {line["name"] for line in lines}
    for old_name in list(saved):
        if old_name not in current_names:
            saved.pop(old_name, None)
            counts.pop(old_name, None)
            changed.append(old_name + " (removed)")
    for line in lines:
        name = line["name"]
        current = line_signature(line)
        previous = saved.get(name)
        if previous is not None and previous != current:
            counts.pop(name, None)
            changed.append(name)
        saved[name] = current
    return changed


def save_progress(path, progress):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

def write_outputs(video_stem, lines, segments, progress):
    GT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = GT_DIR / f"{video_stem}_manual_counts_by_line.csv"
    final_path = GT_DIR / f"{video_stem}_manual_counts.csv"
    minute_path = GT_DIR / f"{video_stem}_manual_counts_by_minute.csv"

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["line_id", "start_x", "start_y", "end_x", "end_y", "start_sec", "end_sec", *CLASSES])
        for line in lines:
            line_counts = progress.get("counts", {}).get(line["name"], {})
            for start, end in segments:
                item = line_counts.get(segment_key(start, end))
                if item is not None:
                    writer.writerow([
                        line["name"], line["start"][0], line["start"][1],
                        line["end"][0], line["end"][1],
                        start, end, *[item[c] for c in CLASSES],
                    ])

    complete_rows = []
    for start, end in segments:
        key = segment_key(start, end)
        items = [progress.get("counts", {}).get(line["name"], {}).get(key) for line in lines]
        if all(item is not None for item in items):
            totals = {c: sum(item[c] for item in items) for c in CLASSES}
            complete_rows.append((start, end, totals))

    with final_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", *CLASSES])
        for start, end, totals in complete_rows:
            writer.writerow([start, end, *[totals[c] for c in CLASSES]])

    complete_map = {(start, end): totals for start, end, totals in complete_rows}
    minute_groups = {}
    for start, end in segments:
        minute = int(start // 60)
        minute_groups.setdefault(minute, []).append((start, end))

    with minute_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["minute", *CLASSES])
        for minute, expected in sorted(minute_groups.items()):
            minute_end = (minute + 1) * 60
            if any(end > minute_end + 1e-6 for _, end in expected):
                continue
            if not all(pair in complete_map for pair in expected):
                continue
            totals = {c: sum(complete_map[pair][c] for pair in expected) for c in CLASSES}
            writer.writerow([minute, *[totals[c] for c in CLASSES]])

    return detail_path, final_path, minute_path


def fit_frame(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def draw_line(frame, line):
    canvas = frame.copy()
    p1 = tuple(map(int, line["start"]))
    p2 = tuple(map(int, line["end"]))
    cv2.line(canvas, p1, p2, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.circle(canvas, p1, 9, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, p2, 9, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas

def play_segment(video_path, line, start, end, info, speed=1.0):
    cap = cv2.VideoCapture(str(video_path))
    fps = info["fps"]
    start_frame = int(round(start * fps))
    end_frame = min(info["frame_count"], int(round(end * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    delay = max(1, int(round(1000.0 / (fps * speed))))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    frame_index = start_frame
    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = draw_line(frame, line)
        current_sec = frame_index / fps
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 82), (0, 0, 0), -1)
        cv2.putText(canvas, f"{line['name']} | START {tuple(line['start'])} -> END {tuple(line['end'])}",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Segment {start:.2f}s - {end:.2f}s | now {current_sec:.2f}s",
                    (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, fit_frame(canvas))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            cap.release()
            cv2.destroyWindow(WINDOW_NAME)
            return "quit"
        if key in (ord("r"), ord("R")):
            cap.release()
            cv2.destroyWindow(WINDOW_NAME)
            return "replay"
        if key == ord(" "):
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord(" "):
                    break
                if pause_key in (ord("r"), ord("R")):
                    cap.release()
                    cv2.destroyWindow(WINDOW_NAME)
                    return "replay"
                if pause_key in (ord("q"), ord("Q"), 27):
                    cap.release()
                    cv2.destroyWindow(WINDOW_NAME)
                    return "quit"
        frame_index += 1

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)
    return "done"


def read_counts():
    print("\nNhập 5 số theo đúng thứ tự:")
    print("  motorcycle car bus truck bicycle")
    while True:
        raw = input("> ").replace(",", " ").split()
        if len(raw) == len(CLASSES) and all(x.isdigit() for x in raw):
            return {c: int(v) for c, v in zip(CLASSES, raw)}
        print("Cần đúng 5 số nguyên >= 0. Ví dụ: 12 4 0 1 0")

def count_segment(video_path, line, start, end, info, speed):
    while True:
        action = play_segment(video_path, line, start, end, info, speed)
        if action == "quit":
            return None, "quit"
        if action == "replay":
            continue

        print(f"\nĐã phát xong {line['name']} | {start:.2f}s -> {end:.2f}s")
        choice = input("Enter = nhập count | R = xem lại | Q = thoát: ").strip().lower()
        if choice == "r":
            continue
        if choice == "q":
            return None, "quit"

        while True:
            counts = read_counts()
            print("\nBạn vừa nhập:")
            print("  " + " | ".join(f"{c}={counts[c]}" for c in CLASSES))
            choice = input("Enter = xác nhận | E = nhập lại | V = xem lại video | Q = thoát: ").strip().lower()
            if choice == "e":
                continue
            if choice == "v":
                break
            if choice == "q":
                return None, "quit"
            return counts, "confirmed"

def interval_label(interval):
    if abs(interval - round(interval)) < 1e-9:
        return f"{int(round(interval))}s"
    return (f"{interval:g}s").replace(".", "p")


def write_all_outputs(video_stem, lines, segments, progress, interval):
    detail, aggregate, minute = write_outputs(video_stem, lines, segments, progress)
    interval_path = GT_DIR / f"{video_stem}_manual_counts_by_{interval_label(interval)}.csv"
    interval_path.write_bytes(aggregate.read_bytes())
    return {
        "by_line": detail,
        "aggregate": aggregate,
        "by_interval": interval_path,
        "by_minute": minute,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Setup lines and create Manual Ground Truth for any video.")
    parser.add_argument("--video", help="Video path; omit to choose from data/videos interactively")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="Counting segment length in seconds (default: 10)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default: 1.0)")
    parser.add_argument("--edit-lines", action="store_true", help="Force line setup/edit before counting")
    parser.add_argument("--setup-only", action="store_true", help="Only configure lines, do not count")
    parser.add_argument("--resume", action="store_true", help="Resume saved progress without asking")
    parser.add_argument("--restart", action="store_true", help="Discard saved progress and recount")
    return parser.parse_args()


def new_progress(video_id, video_file, interval, line_config_path=None):
    return {
        "video_id": video_id,
        "video": video_file,
        "interval_sec": interval,
        "line_config": str(line_config_path) if line_config_path else None,
        "counts": {},
    }


def choose_resume(progress_path, args):
    if not progress_path.exists():
        return False
    if args.resume:
        return True
    if args.restart:
        return False
    print(f"\nDa co tien do cu: {progress_path}")
    choice = input("Enter = resume | N = dem lai tu dau: ").strip().lower()
    return choice != "n"


def print_outputs(paths):
    print(f"By-line CSV: {paths['by_line']}")
    print(f"Aggregate CSV: {paths['aggregate']}")
    print(f"Interval CSV: {paths['by_interval']}")
    print(f"Minute CSV: {paths['by_minute']}")


def main():
    args = parse_args()
    if args.interval <= 0 or args.speed <= 0:
        raise ValueError("--interval and --speed must be > 0")
    if args.resume and args.restart:
        raise ValueError("Choose only one of --resume or --restart")

    video_path = choose_video_path(args.video)
    video_cfg, config_path = setup_lines(video_path, force_edit=args.edit_lines)
    print(f"\nLine config: {config_path}")
    if args.setup_only:
        print("Setup-only complete.")
        return

    info = get_video_info(video_path)
    lines = video_cfg.get("lines", [])
    if not lines:
        raise ValueError("No counting lines configured")
    segments = make_segments(info["duration"], args.interval)
    video_stem = video_path.stem
    progress_path = GT_DIR / f"{video_stem}_manual_count_progress.json"

    if choose_resume(progress_path, args):
        progress = load_progress(progress_path, video_stem, video_path.name, args.interval)
        old_interval = float(progress.get("interval_sec", args.interval))
        if abs(old_interval - args.interval) > 1e-9:
            raise ValueError(
                f"Saved progress uses interval={old_interval}s, current interval={args.interval}s. "
                "Use the same interval or --restart."
            )
        changed_lines = sync_line_geometry(progress, lines)
    else:
        progress = new_progress(video_stem, video_path.name, args.interval, config_path)
        changed_lines = sync_line_geometry(progress, lines)
        write_all_outputs(video_stem, lines, segments, progress, args.interval)
    progress["line_config"] = str(config_path)
    save_progress(progress_path, progress)

    print("\n=== MANUAL GROUND TRUTH TOOL ===")
    print(f"Video: {video_path}")
    print(f"Duration: {info['duration']:.2f}s | FPS: {info['fps']:.2f}")
    print(f"Interval: {args.interval:.2f}s | Lines: {len(lines)}")
    print("Rule: nhin START -> END, chi dem crossing LEFT -> RIGHT.")
    print("Playback: SPACE pause/resume | R replay | Q/Esc quit")
    if changed_lines:
        print("Line geometry changed, old counts removed for: " + ", ".join(changed_lines))

    total_tasks = len(lines) * len(segments)
    completed = sum(
        1 for line in lines for start, end in segments
        if segment_key(start, end) in progress.get("counts", {}).get(line["name"], {})
    )
    if completed:
        print(f"Progress: {completed}/{total_tasks} tasks already confirmed.")

    for line_index, line in enumerate(lines, 1):
        line_counts = progress.setdefault("counts", {}).setdefault(line["name"], {})
        print("\n" + "=" * 70)
        print(f"LINE {line_index}/{len(lines)}: {line['name']}")
        print(f"START {line['start']} -> END {line['end']}")
        for segment_index, (start_sec, end_sec) in enumerate(segments, 1):
            key = segment_key(start_sec, end_sec)
            if key in line_counts:
                print(f"  [{segment_index}/{len(segments)}] {start_sec:.2f}-{end_sec:.2f}s: done")
                continue
            print(f"\n[{segment_index}/{len(segments)}] {start_sec:.2f}s -> {end_sec:.2f}s")
            counts, status = count_segment(
                video_path, line, start_sec, end_sec, info, args.speed
            )
            if status == "quit":
                save_progress(progress_path, progress)
                paths = write_all_outputs(video_stem, lines, segments, progress, args.interval)
                print("\nDa luu tien do truoc khi thoat.")
                print(f"Progress JSON: {progress_path}")
                print_outputs(paths)
                return
            line_counts[key] = counts
            save_progress(progress_path, progress)
            write_all_outputs(video_stem, lines, segments, progress, args.interval)
            print("Da luu doan nay.")

    paths = write_all_outputs(video_stem, lines, segments, progress, args.interval)
    save_progress(progress_path, progress)
    cv2.destroyAllWindows()
    print("\n" + "=" * 70)
    print("HOAN TAT MANUAL GROUND TRUTH")
    print(f"Progress JSON: {progress_path}")
    print(f"Line config: {config_path}")
    print_outputs(paths)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nDa dung.")
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        cv2.destroyAllWindows()
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
