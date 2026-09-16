"""OpenCV overlay for multi-line vehicle counting."""

from __future__ import annotations

from collections import Counter

import cv2

from src.counting.counter import normal_direction


LINE_COLORS = [(70, 220, 255), (255, 130, 70), (190, 100, 255), (80, 220, 120)]


def _text(frame, value: str, point: tuple[int, int], color, scale=0.55, thickness=2):
    cv2.putText(
        frame,
        str(value),
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_counting_lines(frame, lines, arrow_length: int = 70) -> None:
    """Draw each line and perpendicular IN/OUT arrows from its midpoint."""

    for index, line in enumerate(lines):
        color = LINE_COLORS[index % len(LINE_COLORS)]
        cv2.line(frame, line.start, line.end, color, 3, cv2.LINE_AA)
        _text(frame, line.name, (line.start[0], max(22, line.start[1] - 10)), color, 0.65)

        mid_x = (line.start[0] + line.end[0]) // 2
        mid_y = (line.start[1] + line.end[1]) // 2
        if line.direction_vector is None:
            vector_x, vector_y = normal_direction(line.start, line.end)
        else:
            vector_x, vector_y = line.direction_vector
            length = max((vector_x**2 + vector_y**2) ** 0.5, 1e-9)
            vector_x, vector_y = vector_x / length, vector_y / length

        in_tip = (int(mid_x + vector_x * arrow_length), int(mid_y + vector_y * arrow_length))
        out_tip = (int(mid_x - vector_x * arrow_length), int(mid_y - vector_y * arrow_length))
        cv2.arrowedLine(frame, (mid_x, mid_y), in_tip, (50, 230, 50), 3, tipLength=0.25)
        cv2.arrowedLine(frame, (mid_x, mid_y), out_tip, (40, 100, 255), 3, tipLength=0.25)
        _text(frame, line.in_label, (in_tip[0] + 5, in_tip[1] - 5), (50, 230, 50), 0.62)
        _text(frame, line.out_label, (out_tip[0] + 5, out_tip[1] - 5), (40, 100, 255), 0.62)


def draw_vehicles(frame, vehicles) -> None:
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 235, 0), 2)
        cv2.circle(frame, vehicle.center, 4, (0, 255, 255), -1)
        _text(
            frame,
            f"{vehicle.class_name} #{vehicle.track_id} {vehicle.confidence:.2f}",
            (x1, max(18, y1 - 7)),
            (0, 235, 0),
            0.48,
            1,
        )


def draw_statistics(frame, counts: Counter, total: int) -> None:
    rows = [f"TOTAL: {total}"]
    for (line, class_name, direction), count in sorted(counts.items()):
        rows.append(f"{line} | {direction} | {class_name}: {count}")
    panel_width = min(frame.shape[1] - 20, 500)
    panel_height = min(frame.shape[0] - 20, 20 + 25 * len(rows))
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for index, row in enumerate(rows):
        _text(frame, row, (20, 35 + 24 * index), (255, 255, 255), 0.55, 1)


def render_frame(frame, lines, vehicles, counts: Counter, total: int):
    draw_counting_lines(frame, lines)
    draw_vehicles(frame, vehicles)
    draw_statistics(frame, counts, total)
    return frame
