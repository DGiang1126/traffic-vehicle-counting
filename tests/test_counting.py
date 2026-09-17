import unittest

from src.counting import LineDefinition, MultiLineCounter, normal_direction


class MultiLineCounterTests(unittest.TestCase):
    def setUp(self):
        line = LineDefinition(
            name="line_1",
            start=(0, 100),
            end=(200, 100),
            negative_to_positive="top_to_bottom",
            positive_to_negative="bottom_to_top",
        )
        self.counter = MultiLineCounter([line])

    def update(self, track_id, center, frame_index):
        return self.counter.update(
            track_id=track_id,
            center=center,
            class_name="car",
            confidence=0.9,
            timestamp=frame_index / 10,
            frame_index=frame_index,
        )

    def test_top_to_bottom_crossing(self):
        self.assertEqual(self.update(1, (50, 90), 0), [])
        events = self.update(1, (50, 110), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, "top_to_bottom")

    def test_bottom_to_top_crossing(self):
        self.assertEqual(self.update(2, (50, 110), 0), [])
        events = self.update(2, (50, 90), 1)
        self.assertEqual(events[0].direction, "bottom_to_top")

    def test_same_track_is_not_counted_twice_on_same_line(self):
        self.update(3, (50, 90), 0)
        self.assertEqual(len(self.update(3, (50, 110), 1)), 1)
        self.update(3, (50, 90), 2)
        self.assertEqual(self.update(3, (50, 110), 3), [])

    def test_touching_line_keeps_previous_side(self):
        self.update(4, (50, 90), 0)
        self.assertEqual(self.update(4, (50, 100), 1), [])
        self.assertEqual(len(self.update(4, (50, 110), 2)), 1)

    def test_crossing_infinite_extension_is_not_counted(self):
        self.update(5, (250, 90), 0)
        self.assertEqual(self.update(5, (250, 110), 1), [])

    def test_same_track_can_be_counted_once_on_each_line(self):
        second_line = LineDefinition(
            name="line_2",
            start=(0, 150),
            end=(200, 150),
            negative_to_positive="top_to_bottom",
            positive_to_negative="bottom_to_top",
        )
        counter = MultiLineCounter([self.counter.lines[0], second_line])

        common = {
            "track_id": 6,
            "class_name": "car",
            "confidence": 0.9,
        }
        self.assertEqual(
            counter.update(center=(50, 90), timestamp=0, frame_index=0, **common), []
        )
        first_events = counter.update(
            center=(50, 110), timestamp=0.1, frame_index=1, **common
        )
        second_events = counter.update(
            center=(50, 160), timestamp=0.2, frame_index=2, **common
        )

        self.assertEqual([event.line for event in first_events], ["line_1"])
        self.assertEqual([event.line for event in second_events], ["line_2"])

    def test_direction_vector_does_not_depend_on_endpoint_order(self):
        common = {
            "name": "entrance",
            "direction_vector": (0, 1),
            "in_label": "IN",
            "out_label": "OUT",
        }
        normal = MultiLineCounter(
            [LineDefinition(start=(0, 100), end=(200, 100), **common)]
        )
        reversed_line = MultiLineCounter(
            [LineDefinition(start=(200, 100), end=(0, 100), **common)]
        )

        for counter in (normal, reversed_line):
            counter.update(
                track_id=10,
                center=(50, 90),
                class_name="car",
                confidence=0.9,
                timestamp=0,
                frame_index=0,
            )
            events = counter.update(
                track_id=10,
                center=(50, 110),
                class_name="car",
                confidence=0.9,
                timestamp=0.1,
                frame_index=1,
            )
            self.assertEqual(events[0].direction, "IN")

    def test_direction_vector_reports_reverse_crossing(self):
        counter = MultiLineCounter(
            [
                LineDefinition(
                    name="entrance",
                    start=(0, 100),
                    end=(200, 100),
                    direction_vector=(0, 1),
                    in_label="IN",
                    out_label="OUT",
                )
            ]
        )
        common = {"track_id": 11, "class_name": "car", "confidence": 0.9}
        counter.update(center=(50, 110), timestamp=0, frame_index=0, **common)
        events = counter.update(
            center=(50, 90), timestamp=0.1, frame_index=1, **common
        )
        self.assertEqual(events[0].direction, "OUT")

    def test_normal_direction_is_perpendicular_and_reversible(self):
        forward = normal_direction((10, 20), (110, 70))
        reverse = normal_direction((10, 20), (110, 70), reverse=True)
        tangent = (100, 50)

        self.assertAlmostEqual(tangent[0] * forward[0] + tangent[1] * forward[1], 0)
        self.assertAlmostEqual(reverse[0], -forward[0])
        self.assertAlmostEqual(reverse[1], -forward[1])


if __name__ == "__main__":
    unittest.main()
