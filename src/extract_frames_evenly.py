import argparse
from pathlib import Path

import cv2
import numpy as np


def extract_frames_evenly(
    video_path: str,
    output_dir: str,
    num_frames: int,
    prefix: str,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise RuntimeError("Không đọc được tổng số frame.")

    if num_frames > total_frames:
        raise ValueError(
            f"Yêu cầu {num_frames} frame nhưng video chỉ có {total_frames} frame."
        )

    # Chọn đúng num_frames vị trí, rải đều từ đầu đến cuối video
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames,
        dtype=int,
    )

    saved = 0

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

        ok, frame = cap.read()

        if not ok:
            print(f"[WARNING] Không đọc được frame {frame_idx}")
            continue

        filename = output_dir / f"{prefix}_f{frame_idx:06d}.jpg"

        success = cv2.imwrite(str(filename), frame)

        if success:
            saved += 1
        else:
            print(f"[WARNING] Không lưu được: {filename}")

    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    print()
    print("===== EXTRACT SUMMARY =====")
    print(f"Video       : {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS         : {fps:.2f}")
    print(f"Duration    : {duration:.2f} s")
    print(f"Requested   : {num_frames}")
    print(f"Saved       : {saved}")
    print(f"Output      : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a fixed number of evenly spaced frames from video."
    )

    parser.add_argument(
        "video",
        help="Đường dẫn video input",
    )

    parser.add_argument(
        "output",
        help="Thư mục lưu ảnh",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Số frame cần extract",
    )

    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix tên ảnh",
    )

    args = parser.parse_args()

    extract_frames_evenly(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.num_frames,
        prefix=args.prefix,
    )