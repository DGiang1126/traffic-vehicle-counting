"""Finite line-crossing counter for the E4 experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

Point = tuple[int, int]


def normal_direction(start: Point, end: Point, reverse: bool = False) -> tuple[float, float]:
    """Return a unit vector perpendicular to a line.

    The vector is used as the semantic IN direction. Reversing it swaps IN and
    OUT without depending on the order of the two line endpoints.
    """

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("A counting line must have two distinct points")
    vector = (-dy / length, dx / length)
    return (-vector[0], -vector[1]) if reverse else vector


@dataclass(frozen=True)
class LineDefinition:
    """A finite counting line and the names of its two crossing directions."""

    name: str
    start: Point
    end: Point
    direction_vector: tuple[float, float] | None = None
    in_label: str = "IN"
    out_label: str = "OUT"
    negative_to_positive: str = "negative_to_positive"
    positive_to_negative: str = "positive_to_negative"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Counting line name must not be empty")
        if self.start == self.end:
            raise ValueError(f"Counting line {self.name!r} must have two distinct points")
        if self.direction_vector is not None:
            dx, dy = self.direction_vector
            if math.hypot(dx, dy) == 0:
                raise ValueError(f"Direction vector for {self.name!r} must not be zero")
        if not self.in_label.strip() or not self.out_label.strip():
            raise ValueError(f"IN/OUT labels for {self.name!r} must not be empty")
        if self.in_label == self.out_label:
            raise ValueError(f"IN and OUT labels for {self.name!r} must be different")


@dataclass(frozen=True)
class CountingEvent:
    timestamp: float
    frame_index: int
    track_id: int
    class_name: str
    line: str
    direction: str
    confidence: float

    def to_dict(self) -> dict:
        row = asdict(self)
        row["class"] = row.pop("class_name")
        return row


def _side(point: Point, line: LineDefinition) -> int:
    x, y = point
    x1, y1 = line.start
    x2, y2 = line.end
    value = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _on_segment(a: Point, b: Point, point: Point) -> bool:
    return (
        min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return False


def _direction_name(
    line: LineDefinition,
    previous_point: Point,
    current_point: Point,
    previous_side: int,
    current_side: int,
) -> str:
    if line.direction_vector is not None:
        move_x = current_point[0] - previous_point[0]
        move_y = current_point[1] - previous_point[1]
        direction_x, direction_y = line.direction_vector
        dot_product = move_x * direction_x + move_y * direction_y
        if dot_product > 0:
            return line.in_label
        if dot_product < 0:
            return line.out_label

    return (
        line.negative_to_positive
        if previous_side < current_side
        else line.positive_to_negative
    )


class MultiLineCounter:
    """Create at most one event for each `(track_id, line_name)` pair."""

    def __init__(self, lines: Iterable[LineDefinition]):
        self.lines = list(lines)
        if not self.lines:
            raise ValueError("At least one counting line is required")
        names = [line.name for line in self.lines]
        if len(names) != len(set(names)):
            raise ValueError("Counting line names must be unique")
        self._last_side: dict[tuple[int, str], int] = {}
        self._last_point: dict[tuple[int, str], Point] = {}
        self._counted: set[tuple[int, str]] = set()

    def update(
        self,
        *,
        track_id: int,
        center: Point,
        class_name: str,
        confidence: float,
        timestamp: float,
        frame_index: int,
    ) -> list[CountingEvent]:
        events: list[CountingEvent] = []

        for line in self.lines:
            key = (int(track_id), line.name)
            current_side = _side(center, line)
            previous_side = self._last_side.get(key)

            # Do not erase the last known side when the center is exactly on a line.
            if current_side == 0:
                continue

            if (
                previous_side is not None
                and previous_side != current_side
                and key not in self._counted
                and _segments_intersect(
                    self._last_point[key], center, line.start, line.end
                )
            ):
                events.append(
                    CountingEvent(
                        timestamp=round(float(timestamp), 3),
                        frame_index=int(frame_index),
                        track_id=int(track_id),
                        class_name=class_name,
                        line=line.name,
                        direction=_direction_name(
                            line,
                            self._last_point[key],
                            center,
                            previous_side,
                            current_side,
                        ),
                        confidence=round(float(confidence), 4),
                    )
                )
                self._counted.add(key)

            self._last_side[key] = current_side
            self._last_point[key] = center

        return events
