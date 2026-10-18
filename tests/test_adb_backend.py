import unittest

import numpy as np

from katacr.policy.env.adb import AdbDevice, normalize_frame
from katacr.policy.env.utils import resolve_tap_xy


class FakeRunner:
  def __init__(self, stdout=b"", returncode=0, stderr=b""):
    self.stdout = stdout
    self.returncode = returncode
    self.stderr = stderr
    self.commands = []

  def __call__(self, command, **kwargs):
    self.commands.append((command, kwargs))
    return type("Result", (), {
      "stdout": self.stdout,
      "stderr": self.stderr,
      "returncode": self.returncode,
    })()


class AdbBackendTest(unittest.TestCase):
  def test_command_includes_serial(self):
    runner = FakeRunner()
    device = AdbDevice(adb="adb-test", serial="device-1", runner=runner)

    device.tap(12.8, 34.2)

    self.assertEqual(
      runner.commands[0][0],
      ["adb-test", "-s", "device-1", "shell", "input", "tap", "12", "34"],
    )

  def test_resize_width_returns_reverse_scale(self):
    frame = np.zeros((240, 160, 3), dtype=np.uint8)

    normalized, offset, scale = normalize_frame(frame, target_ratio=2.0, mode="resize_width")

    self.assertEqual(normalized.shape[:2], (240, 120))
    self.assertEqual(offset, (0, 0))
    self.assertEqual(scale, (160 / 120, 1))

  def test_center_crop_returns_reverse_offset(self):
    frame = np.zeros((240, 160, 3), dtype=np.uint8)

    normalized, offset, scale = normalize_frame(frame, target_ratio=2.0, mode="center_crop")

    self.assertEqual(normalized.shape[:2], (240, 120))
    self.assertEqual(offset, (20, 0))
    self.assertEqual(scale, (1, 1))

  def test_tap_coordinates_apply_transform_and_clamp(self):
    xy = resolve_tap_xy(
      (0.75, 0.5),
      img_size=(120, 240),
      offset=(20, 0),
      scale=(1, 1),
      clamp=(0, 0, 100, 200),
    )

    self.assertEqual(xy, (100, 120))


if __name__ == "__main__":
  unittest.main()
