"""Minimal Counting demo that does not need YOLO, OpenCV or a video."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.counting import LineDefinition, MultiLineCounter


def main() -> None:
    line = LineDefinition(
        name="gate_main",
        start=(0, 100),
        end=(200, 100),
        negative_to_positive="top_to_bottom",
        positive_to_negative="bottom_to_top",
    )
    counter = MultiLineCounter([line])

    centers = [(80, 80), (82, 92), (84, 108), (86, 120)]
    for frame_index, center in enumerate(centers):
        events = counter.update(
            track_id=101,
            center=center,
            class_name="car",
            confidence=0.92,
            timestamp=frame_index / 10,
            frame_index=frame_index,
        )
        for event in events:
            print(event.to_dict())


if __name__ == "__main__":
    main()
