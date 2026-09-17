"""Finite line-crossing counter for the E4 experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


Point = tuple[int, int]


def normal_direction(
    start: Point,
    end: Point,
    reverse: bool = False,
) -> tuple[float, float]:
    """Return a unit vector perpendicular to a line.

    The vector is used as the semantic IN direction.
    Reversing it swaps IN and OUT without depending
    on the order of the two line endpoints.
    """

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = math.hypot(dx, dy)

    if length == 0:
        raise ValueError(
            "A counting line must have two distinct points"
        )

    vector = (
        -dy / length,
        dx / length,
    )

    if reverse:
        return (
            -vector[0],
            -vector[1],
        )

    return vector


@dataclass(frozen=True)
class LineDefinition:
    """A finite counting line and its crossing directions."""

    name: str
    start: Point
    end: Point

    # Optional semantic direction.
    #
    # If direction_vector is supplied:
    #   movement along vector  -> in_label
    #   movement opposite      -> out_label
    #
    # If direction_vector is None:
    #   signed-side direction is used instead.
    direction_vector: tuple[float, float] | None = None

    in_label: str = "IN"
    out_label: str = "OUT"

    negative_to_positive: str = "negative_to_positive"
    positive_to_negative: str = "positive_to_negative"

    def __post_init__(self) -> None:

        if not self.name.strip():
            raise ValueError(
                "Counting line name must not be empty"
            )

        if self.start == self.end:
            raise ValueError(
                f"Counting line {self.name!r} "
                "must have two distinct points"
            )

        if self.direction_vector is not None:
            dx, dy = self.direction_vector

            if math.hypot(dx, dy) == 0:
                raise ValueError(
                    f"Direction vector for "
                    f"{self.name!r} must not be zero"
                )

        if (
            not self.in_label.strip()
            or not self.out_label.strip()
        ):
            raise ValueError(
                f"IN/OUT labels for "
                f"{self.name!r} must not be empty"
            )

        if self.in_label == self.out_label:
            raise ValueError(
                f"IN and OUT labels for "
                f"{self.name!r} must be different"
            )


@dataclass(frozen=True)
class CountingEvent:
    """One valid vehicle crossing event."""

    timestamp: float
    frame_index: int
    track_id: int
    class_name: str
    line: str
    direction: str
    confidence: float

    def to_dict(self) -> dict:
        row = asdict(self)

        # Statistics module expects column name "class".
        row["class"] = row.pop("class_name")

        return row


def _side(
    point: Point,
    line: LineDefinition,
) -> int:
    """Return which side of the directed line a point lies on.

    Returns:
         1 -> positive side
        -1 -> negative side
         0 -> exactly on the line
    """

    x, y = point

    x1, y1 = line.start
    x2, y2 = line.end

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def _orientation(
    a: Point,
    b: Point,
    c: Point,
) -> int:
    """Orientation used by finite-segment intersection."""

    value = (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def _on_segment(
    a: Point,
    b: Point,
    point: Point,
) -> bool:
    """Check whether point lies inside finite segment a-b."""

    return (
        min(a[0], b[0])
        <= point[0]
        <= max(a[0], b[0])
        and min(a[1], b[1])
        <= point[1]
        <= max(a[1], b[1])
    )


def _segments_intersect(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
) -> bool:
    """Check whether finite segments a-b and c-d intersect."""

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    # General case.
    if o1 != o2 and o3 != o4:
        return True

    # Collinear / touching cases.
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
    """Determine crossing direction name."""

    # ======================================================
    # SEMANTIC IN / OUT DIRECTION
    # ======================================================

    if line.direction_vector is not None:

        move_x = (
            current_point[0]
            - previous_point[0]
        )

        move_y = (
            current_point[1]
            - previous_point[1]
        )

        direction_x, direction_y = (
            line.direction_vector
        )

        dot_product = (
            move_x * direction_x
            + move_y * direction_y
        )

        if dot_product > 0:
            return line.in_label

        if dot_product < 0:
            return line.out_label

    # ======================================================
    # LEGACY SIGNED-SIDE DIRECTION
    # ======================================================

    if previous_side < current_side:
        return line.negative_to_positive

    return line.positive_to_negative


class MultiLineCounter:
    """Count crossings for one or more finite counting lines.

    A track may be counted at most once on each line.

    Example:
        track_id=5 crossing line_1 -> counted once
        track_id=5 crossing line_1 again -> ignored
        track_id=5 crossing line_2 -> counted once

    allowed_direction:
        None
            Count both crossing directions.

        "negative_to_positive"
            Count only negative -> positive crossings.

        "positive_to_negative"
            Count only positive -> negative crossings.

        "IN" / "OUT"
            Can also be used when LineDefinition has a
            direction_vector and corresponding labels.
    """

    def __init__(
        self,
        lines: Iterable[LineDefinition],
        allowed_direction: str | None = None,
    ):
        self.lines = list(lines)

        self.allowed_direction = allowed_direction

        if not self.lines:
            raise ValueError(
                "At least one counting line is required"
            )

        names = [
            line.name
            for line in self.lines
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Counting line names must be unique"
            )

        # Last non-zero side of each track for each line.
        self._last_side: dict[
            tuple[int, str],
            int,
        ] = {}

        # Last known center point.
        self._last_point: dict[
            tuple[int, str],
            Point,
        ] = {}

        # Already counted (track_id, line_name).
        self._counted: set[
            tuple[int, str]
        ] = set()

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
        """Update counter using one tracked vehicle observation."""

        events: list[CountingEvent] = []

        # ==================================================
        # CHECK EVERY COUNTING LINE
        # ==================================================

        for line in self.lines:

            key = (
                int(track_id),
                line.name,
            )

            current_side = _side(
                center,
                line,
            )

            previous_side = (
                self._last_side.get(key)
            )

            # --------------------------------------------------
            # CENTER IS EXACTLY ON LINE
            # --------------------------------------------------
            #
            # Do not overwrite previous side.
            #
            # Example:
            # frame 1: negative
            # frame 2: exactly on line
            # frame 3: positive
            #
            # We still want to detect negative -> positive.
            # --------------------------------------------------

            if current_side == 0:
                continue

            # ==================================================
            # POSSIBLE CROSSING
            # ==================================================

            if (
                previous_side is not None
                and previous_side != current_side
                and key not in self._counted
                and _segments_intersect(
                    self._last_point[key],
                    center,
                    line.start,
                    line.end,
                )
            ):

                # ----------------------------------------------
                # DETERMINE DIRECTION
                # ----------------------------------------------

                direction = _direction_name(
                    line,
                    self._last_point[key],
                    center,
                    previous_side,
                    current_side,
                )

                # ----------------------------------------------
                # OPTIONAL DIRECTION FILTER
                # ----------------------------------------------
                #
                # Important:
                # If direction is not accepted, DO NOT add the
                # track to self._counted.
                #
                # This allows the same track to later cross the
                # line in the valid direction.
                # ----------------------------------------------

                if (
                    self.allowed_direction
                    is not None
                    and direction
                    != self.allowed_direction
                ):
                    self._last_side[key] = (
                        current_side
                    )

                    self._last_point[key] = (
                        center
                    )

                    continue

                # ----------------------------------------------
                # CREATE VALID COUNTING EVENT
                # ----------------------------------------------

                event = CountingEvent(
                    timestamp=round(
                        float(timestamp),
                        3,
                    ),
                    frame_index=int(
                        frame_index
                    ),
                    track_id=int(
                        track_id
                    ),
                    class_name=class_name,
                    line=line.name,
                    direction=direction,
                    confidence=round(
                        float(confidence),
                        4,
                    ),
                )

                events.append(event)

                # Only mark as counted after a VALID event.
                self._counted.add(key)

            # ==================================================
            # UPDATE TRACK STATE
            # ==================================================

            self._last_side[key] = (
                current_side
            )

            self._last_point[key] = (
                center
            )

        return events