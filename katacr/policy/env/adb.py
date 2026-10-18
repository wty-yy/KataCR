import os
import subprocess

import cv2
import numpy as np


class AdbDevice:
  """Small ADB backend for screenshots and coordinate-based input."""

  def __init__(self, adb=None, serial=None, timeout=10, runner=None):
    self.adb = adb or os.getenv("KATACR_ADB", "adb")
    self.serial = serial if serial is not None else os.getenv("KATACR_ADB_SERIAL")
    self.timeout = timeout
    self.runner = runner or subprocess.run

  def _command(self, *args):
    command = [self.adb]
    if self.serial:
      command += ["-s", self.serial]
    return command + list(args)

  def _run(self, *args):
    command = self._command(*args)
    result = self.runner(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      timeout=self.timeout,
    )
    if result.returncode != 0:
      stderr = result.stderr.decode("utf-8", errors="replace").strip()
      raise RuntimeError(f"ADB command failed ({result.returncode}): {stderr}")
    return result.stdout

  def screencap(self):
    data = self._run("exec-out", "screencap", "-p")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
      raise RuntimeError("ADB returned an invalid PNG screenshot")
    return image

  def tap(self, x, y):
    self._run("shell", "input", "tap", str(int(x)), str(int(y)))


def normalize_frame(frame, target_ratio=None, mode="resize_width"):
  """Normalize a portrait frame and return (frame, offset, scale).

  ``target_ratio`` is height / width. The returned offset and scale map
  coordinates in the normalized frame back to the original device frame.
  """
  if target_ratio is None:
    return frame, (0, 0), (1, 1)
  if target_ratio <= 0:
    raise ValueError(f"target_ratio must be positive, got {target_ratio}")
  if mode not in {"resize_width", "center_crop"}:
    raise ValueError(f"unknown frame normalization mode: {mode}")

  height, width = frame.shape[:2]
  target_width = max(1, int(round(height / target_ratio)))
  if target_width >= width:
    return frame, (0, 0), (1, 1)

  if mode == "resize_width":
    normalized = cv2.resize(frame, (target_width, height), interpolation=cv2.INTER_AREA)
    return normalized, (0, 0), (width / target_width, 1)

  crop_x0 = (width - target_width) // 2
  normalized = frame[:, crop_x0:crop_x0 + target_width]
  return normalized, (crop_x0, 0), (1, 1)
