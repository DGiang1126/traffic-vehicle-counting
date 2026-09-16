import sys
import tempfile
import types
import unittest
from pathlib import Path

# Naming helpers do not use OpenCV; this stub lets the test run in lightweight CI.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from src.traffic_counting import _code_version, _unique_run_name


class OutputNamingTest(unittest.TestCase):
    def test_unused_name_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_unique_run_name(Path(directory), "traffic_run"), "traffic_run")

    def test_existing_name_gets_incrementing_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "videos").mkdir()
            (root / "videos" / "traffic_run.mp4").touch()
            self.assertEqual(_unique_run_name(root, "traffic_run"), "traffic_run_02")
            (root / "csv" / "traffic_run_02").mkdir(parents=True)
            self.assertEqual(_unique_run_name(root, "traffic_run"), "traffic_run_03")

    def test_code_version_is_short_sha256(self):
        version = _code_version()
        self.assertEqual(len(version), 12)
        self.assertTrue(all(character in "0123456789abcdef" for character in version))


if __name__ == "__main__":
    unittest.main()

