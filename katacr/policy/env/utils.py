import subprocess, time

def tap_screen(xy, relative=True, img_size=None, delay=0):
  # print("tap:", xy)
  if relative:
    w, h = img_size
    xy = xy[0] * w, xy[1] * h
  subprocess.run(['adb', 'shell', 'input', 'tap', *[str(int(x)) for x in xy]])
  if delay > 0:
    time.sleep(delay)
