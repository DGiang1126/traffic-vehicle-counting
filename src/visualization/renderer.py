"""Draw counting lines, tracked vehicles and live counts on video frames."""

from __future__ import annotations

from collections import Counter
import math

import cv2

LINE_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
]


def draw_counting_lines(frame, lines) -> None:
    for index, line in enumerate(lines):
        color = LINE_COLORS[index % len(LINE_COLORS)]
        cv2.line(frame, line.start, line.end, color, 2)
        cv2.circle(frame, line.start, 5, color, -1)
        cv2.circle(frame, line.end, 5, color, -1)
        cv2.putText(
            frame,
            line.name,
            (line.start[0] + 5, max(20, line.start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        if line.direction_vector is not None:
            direction_x, direction_y = line.direction_vector
            length = math.hypot(direction_x, direction_y)
            if length > 0:
                direction_x /= length
                direction_y /= length
                height, width = frame.shape[:2]
                arrow_length = max(45, min(width, height) // 10)
                middle = (
                    (line.start[0] + line.end[0]) // 2,
                    (line.start[1] + line.end[1]) // 2,
                )
                in_tip = (
                    int(middle[0] + direction_x * arrow_length),
                    int(middle[1] + direction_y * arrow_length),
                )
                out_tip = (
                    int(middle[0] - direction_x * arrow_length),
                    int(middle[1] - direction_y * arrow_length),
                )
                cv2.arrowedLine(
                    frame, middle, in_tip, (0, 255, 0), 3, tipLength=0.25
                )
                cv2.putText(
                    frame,
                    line.forward_direction,
                    (in_tip[0] + 5, max(20, in_tip[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    line.reverse_direction,
                    (out_tip[0] + 5, max(20, out_tip[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )


def draw_vehicles(frame, vehicles) -> None:
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, vehicle.center, 4, (0, 255, 255), -1)
        label = f"{vehicle.class_name} #{vehicle.track_id} {vehicle.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def draw_statistics(frame, event_counts: Counter) -> None:
    total = sum(event_counts.values())
    visible_rows = min(len(event_counts), 12)
    panel_height = max(48, 38 + 23 * visible_rows)
    cv2.rectangle(frame, (5, 5), (430, panel_height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Total line crossings: {total}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    for index, ((line, class_name, direction), count) in enumerate(
        sorted(event_counts.items())[:visible_rows], start=1
    ):
        text = f"{line} | {class_name} | {direction}: {count}"
        cv2.putText(
            frame,
            text,
            (12, 29 + 23 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 0),
            1,
        )


def draw_overlay(frame, vehicles, lines, event_counts: Counter) -> None:
    """Draw all runtime information in-place on one OpenCV frame."""

    draw_counting_lines(frame, lines)
    draw_vehicles(frame, vehicles)
    draw_statistics(frame, event_counts)
