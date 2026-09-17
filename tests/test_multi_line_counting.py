import unittest

from src.counting.counter import LineDefinition, MultiLineCounter, normal_direction


class MultiLineCounterTest(unittest.TestCase):
    def make_counter(self, lines=None):
        return MultiLineCounter(
            lines
            or [
                LineDefinition(
                    "gate", (0, 50), (100, 50),
                    direction_vector=(0, 1), in_label="VAO", out_label="RA",
                )
            ]
        )

    def update(self, counter, track_id, center, frame):
        return counter.update(
            track_id=track_id,
            center=center,
            class_name="car",
            confidence=0.9,
            timestamp=frame / 10,
            frame_index=frame,
        )

    def test_crossing_in_uses_custom_label(self):
        counter = self.make_counter()
        self.update(counter, 1, (30, 40), 1)
        events = self.update(counter, 1, (30, 60), 2)
        self.assertEqual(events[0].direction, "VAO")
        self.assertEqual(events[0].line, "gate")

    def test_crossing_out(self):
        counter = self.make_counter()
        self.update(counter, 1, (30, 60), 1)
        events = self.update(counter, 1, (30, 40), 2)
        self.assertEqual(events[0].direction, "RA")

    def test_same_track_counted_once_on_same_line(self):
        counter = self.make_counter()
        total = []
        for frame, point in enumerate([(30, 40), (30, 60), (30, 40), (30, 60)]):
            total += self.update(counter, 1, point, frame)
        self.assertEqual(len(total), 1)

    def test_same_track_can_count_on_two_lines(self):
        lines = [
            LineDefinition("first", (0, 30), (100, 30), direction_vector=(0, 1)),
            LineDefinition("second", (0, 70), (100, 70), direction_vector=(0, 1)),
        ]
        counter = self.make_counter(lines)
        total = []
        for frame, point in enumerate([(50, 20), (50, 40), (50, 80)]):
            total += self.update(counter, 2, point, frame)
        self.assertEqual({event.line for event in total}, {"first", "second"})

    def test_does_not_count_outside_finite_segment(self):
        counter = self.make_counter()
        self.update(counter, 1, (130, 40), 1)
        events = self.update(counter, 1, (130, 60), 2)
        self.assertEqual(events, [])

    def test_duplicate_line_names_rejected(self):
        with self.assertRaises(ValueError):
            self.make_counter([
                LineDefinition("same", (0, 10), (20, 10)),
                LineDefinition("same", (0, 20), (20, 20)),
            ])

    def test_equal_labels_rejected(self):
        with self.assertRaises(ValueError):
            LineDefinition("gate", (0, 1), (2, 1), in_label="X", out_label="X")

    def test_normal_direction_is_perpendicular(self):
        nx, ny = normal_direction((10, 20), (30, 50))
        self.assertAlmostEqual(20 * nx + 30 * ny, 0.0)

    def test_reversing_normal_swaps_sign(self):
        normal = normal_direction((0, 0), (10, 0))
        reversed_normal = normal_direction((0, 0), (10, 0), reverse=True)
        self.assertEqual(reversed_normal, (-normal[0], -normal[1]))


if __name__ == "__main__":
    unittest.main()
