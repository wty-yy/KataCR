from PIL import Image
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# img = Image.open('/home/wty/Coding/datasets/CR/images/segment/king-tower/king-tower_1_attack_1024.png')
# img = Image.open('/home/wty/Coding/datasets/CR/images/segment/barbarian/barbarian_1_attack_40.png')
img = Image.open('/home/wty/Coding/datasets/CR/images/segment/clock/clock_0_13.png')
# img = Image.open('/home/wty/Coding/datasets/CR/images/segment/the-log/the-log_0_attack_5.png')
# img = Image.open('/home/wty/Coding/datasets/CR/images/segment/backgrounds/background01.jpg')
# img = Image.open('/home/wty/Coding/datasets/CR/images/part2/OYASSU_20210528_episodes/1/00000.jpg')

color2RGB = {
  'red': (255, 0, 0),
  'blue': (0, 0, 255),
  'golden': (255, 215, 0),
  'white': (255, 255, 255),
  'violet': (127, 0, 255),
}
color2alpha = {
  'red': 80,
  'blue': 100,
  'golden': 150,
  'white': 150,
  'violet': 150,
}
color2bright = {
  'red': (30, 50),
  'blue': (30, 80),
  'golden': (70, 80),
  'white': (110, 120),
  'violet': (10, 30),
}
# color2RGBA = {key: val+(255,) for key, val in color2RGB.items()}

def add_filter(
    img: Image.Image | np.ndarray,
    color: str,
    alpha: float = 100,
    bright: float = 0,
    xyxy: Tuple[int] | None = None,
    replace=True
  ):
  if not replace: img = img.copy()
  assert color in color2RGB.keys()
  rgba = color2RGB[color] + (alpha,)
  if not isinstance(img, np.ndarray):
    img = np.array(img)
  # print("Alpha:", img[...,3].max(), img[...,3].min())
  org_bright = img[...,:3].mean()
  assert img.dtype == np.uint8
  if xyxy is None: xyxy = (0, 0, img.shape[1], img.shape[0])
  proc_img = img
  img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
  filter = np.stack([np.full(img.shape[:2], rgba[i], np.uint8) for i in range(4)], -1)
  if img.shape[-1] == 4:
    filter[...,3][img[...,3] == 0] = 0
    proc_img = proc_img[...,:3]
  filter = Image.fromarray(filter)
  img = Image.fromarray(img).convert('RGBA')
  img = np.array(Image.alpha_composite(img, filter).convert('RGB'))
  proc_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] = img
  delta_bright = org_bright - proc_img[...,:3].mean() + bright
  proc_img = (proc_img.astype(np.int32) + delta_bright).clip(0, 255).astype(np.uint8)
  return proc_img

import random
# color = 'blue'
n = len(color2alpha)
for r in range(4):
  for i, color in enumerate(color2bright.keys()):
    plt.subplot(4,n,r*n+i+1)
    bmin, bmax = color2bright[color]
    b = bmin + (bmax - bmin) * (r + 1) / 4
    # img = add_filter(img, 'red', alpha=80, xyxy=(0,56,568,490))
    # print(b)
    new_img = add_filter(img, color, color2alpha[color], bright=b, replace=False)
    plt.title(color + f"({b}b)")
    plt.axis('off')
    plt.imshow(new_img)
plt.tight_layout()
plt.show()
# img = add_filter(img, 'blue', alpha=80, bright=50)
# img = add_filter(img, 'golden', alpha=150, bright=30)
# img = add_filter(img, 'white', alpha=150, bright=50)
# img = Image.fromarray(img)
# img.show()