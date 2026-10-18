import time

from katacr.policy.env.adb import AdbDevice

def resolve_tap_xy(xy, relative=True, img_size=None, offset=(0, 0), scale=(1, 1), clamp=None):
  if relative:
    if img_size is None:
      raise ValueError("img_size is required when relative=True")
    w, h = img_size
    xy = xy[0] * w, xy[1] * h
  ox, oy = offset
  sx, sy = scale
  xy = xy[0] * sx + ox, xy[1] * sy + oy
  if clamp is not None:
    x0, y0, x1, y1 = clamp
    xy = min(max(xy[0], x0), x1), min(max(xy[1], y0), y1)
  return tuple(int(x) for x in xy)


def tap_screen(
    xy,
    relative=True,
    img_size=None,
    delay=0,
    offset=(0, 0),
    scale=(1, 1),
    clamp=None,
    adb=None,
    serial=None,
    device=None,
):
  raw_xy = resolve_tap_xy(xy, relative, img_size, offset, scale, clamp)
  device = device or AdbDevice(adb=adb, serial=serial)
  device.tap(*raw_xy)
  if delay > 0:
    time.sleep(delay)
