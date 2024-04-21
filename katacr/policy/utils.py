import cv2
import numpy as np

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

def pixel2cell(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy - xyxy_grids[:2]) / cell_size

def cell2pixel(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

def xyxy2center(xyxy):
  return np.array([(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2], np.float32)

def xyxy2sub(xyxy, sub):
  """
  'sub' is the sub-xyxy postion relative to xyxy
  """
  w, h = xyxy[2:] - xyxy[:2]
  if not isinstance(sub, np.ndarray):
    sub = np.array(sub)
  delta = sub * np.array([w, h, w, h])
  return np.concatenate([xyxy[:2], xyxy[:2]]) + delta