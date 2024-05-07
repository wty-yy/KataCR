import cv2
import numpy as np

LOW_ALPHA = [chr(ord('a')+i) for i in range(26)]

def extract_img(img, xyxy, target_size=None):
  """
  Args:
    image: The origin image.
    xyxy: The left top and right bottom of the extract image.
    target_size: Resize the extracted image to target size.
  """
  xyxy = np.array(xyxy, np.int32)
  img = img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]]
  if target_size is not None:
    img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
  return img

background_size = (576, 896)  # cell_size = (31.7, 25.1)
xyxy_grids = (4, 82, 574, 884)  # xyxy pixel size of the whold grid
grid_size = (18, 32)  # (x, y)
cell_size = np.array([(xyxy_grids[3] - xyxy_grids[1]) / grid_size[1], (xyxy_grids[2] - xyxy_grids[0]) / grid_size[0]])[::-1]  # cell pixel: (w, h)

def cell2pixel(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

def pixel2cell(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return ((xy - xyxy_grids[:2]) / cell_size).astype(np.float32)

def cell2pixel(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

def xyxy2center(xyxy):
  return np.array([(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2], np.float32)

def xyxy2topcenter(xyxy):
  return np.array([(xyxy[0]+xyxy[2])/2, xyxy[1]], np.float32)

def xyxy2sub(xyxy, sub):
  """
  'sub' is the sub-xyxy postion relative to xyxy
  """
  w, h = xyxy[2:] - xyxy[:2]
  if not isinstance(sub, np.ndarray):
    sub = np.array(sub)
  delta = sub * np.array([w, h, w, h])
  return np.concatenate([xyxy[:2], xyxy[:2]]) + delta

import PIL
from PIL import ImageFont, ImageDraw, Image
from katacr.utils.detection import FONT_PATH
def pil_draw_text(img, xy, text, background_color=(0,0,0), text_color=(255,255,255), font_size=24, text_pos='left top'):
  assert text_pos in ['left top', 'left down', 'right top']
  if isinstance(img, np.ndarray): img = Image.fromarray(img[...,::-1])
  font = ImageFont.truetype(FONT_PATH, font_size)
  pil_version = int(PIL.__version__.split('.')[0])
  xy = np.array(xy)
  texts = text.split('\n')
  if 'down' in text_pos: texts = texts[::-1]
  for t in texts:
    w_text, h_text = font.getbbox(t)[-2:] if pil_version >= 10 else font.getsize(t)
    if text_pos == 'left top':
      x_text, y_text = xy
    elif text_pos == 'left down':
      x_text, y_text = xy[0], xy[1]-h_text
    elif text_pos == 'right top':
      x_text, y_text = xy[0]-w_text, xy[1]
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([x_text, y_text, x_text+w_text, y_text+h_text], radius=1.5, fill=background_color)
    draw.text((x_text, y_text), t, fill=text_color, font=font)
    xy[1] += h_text * (1 if 'top' in text_pos else -1)
  return img

def edit_distance(s1, s2, dis=None):
  """ Levenshtein distance: https://stackoverflow.com/a/24172422 """
  if dis == 's2': s1, s2 = s2, s1
  m=len(s1)+1
  n=len(s2)+1
  s1_dis = max(n, m)  # s1 complete matching
  tbl = {}
  for i in range(m): tbl[i,0]=i
  for j in range(n): tbl[0,j]=j
  for i in range(1, m):
    for j in range(1, n):
      cost = 0 if s1[i-1] == s2[j-1] else 1
      tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)
      if i == m-1:
        s1_dis = min(s1_dis, tbl[i,j])
  if dis == 's1' or dis == 's2':
    return s1_dis
  return tbl[i,j]

if __name__ == '__main__':
  import time
  s1 = 'Lcegc'
  s2 = 'icegolem'

  st = time.time()
  dis = edit_distance(s1, s2)
  print(time.time() - st)
  s1_dis = edit_distance(s1, s2, dis='s1')
  s2_dis = edit_distance(s1, s2, dis='s2')
  print(dis, s1_dis, s2_dis)

