import numpy as np
import cv2
import random
import math
from PIL import Image

def xywh2xyxy(box):
  box[:, 2] = box[:, 0] + box[:, 2]
  box[:, 3] = box[:, 1] + box[:, 3]
  return box

def xywh2cxcywh(box):
  box[:, 0] += box[:, 2] / 2
  box[:, 1] += box[:, 3] / 2
  return box

def xyxy2cxcywh(box):
  box[:, 2] = box[:, 2] - box[:, 0]
  box[:, 3] = box[:, 3] - box[:, 1]
  box[:, 0] += box[:, 2] / 2
  box[:, 1] += box[:, 3] / 2
  return box

def box_filter_idxs(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-6):
  """
  Args:
    box1: Origin box
    box2: New box after transform
    wh_thr: Minimum width and height pixiel threshold
    ar_thr: Maximum aspect ratio threshold
    area_thr: Minimum area box2 / box1 threshold
  """
  w1, h1 = box1[:,2] - box1[:,0], box1[:,3] - box1[:,1]
  w2, h2 = box2[:,2] - box2[:,0], box2[:,3] - box2[:,1]
  ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
  area = w2 * h2 / (w1 * h1 + eps)  # area ratio
  return (w2 > wh_thr) & (h2 > wh_thr) & (area > area_thr) & (ar < ar_thr)

def transform_affine(
    img,
    box=None,       # |   unit       |  random  |   suggestion range |
    rot=0,          # |  degree      |   +/-    |     [0.0, 45.0]    |
    scale=0.5,      # |  wh scale    |   +/-    |     [0.0, 1.0]     |
    shear=0,        # |  wh degree   |   +/-    |     [0, 45]        |
    translate=0.1,  # |  fraction    |   +/-    |     [0, 0.5]       |
    border=0,
    pad_value=(114,114,114),
):
  h = img.shape[0] - border * 2
  w = img.shape[1] - border * 2
  C = np.eye(3)  # Center
  C[0, 2] = -img.shape[1] / 2
  C[1, 2] = -img.shape[0] / 2

  R = np.eye(3)  # Rotation and Scale
  theta = random.uniform(-rot, rot)
  scale = random.uniform(1 - scale, 1 + scale)
  R[0, 0] = R[1, 1] = scale * math.cos(theta * math.pi / 180)
  R[0, 1] = R[1, 0] = scale * math.sin(theta * math.pi / 180)
  R[1, 0] *= -1

  S = np.eye(3)  # Shear
  S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
  S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

  T = np.eye(3)  # Translation
  T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
  T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

  M = T @ S @ R @ C
  img = cv2.warpAffine(img, M[:2], dsize=(w, h), borderValue=pad_value)
  if box is None: return img
  if len(box):
    nb = len(box)
    xy = np.ones((nb * 4, 3))
    xy[:, :2] = box[:, [0,1,2,3,0,3,2,1]].reshape(nb * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(nb, 8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate([x.min(1), y.min(1), x.max(1), y.max(1)]).reshape(4, nb).T
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)
    idxs = box_filter_idxs(box[:, :4] * scale, new)
    box = np.concatenate([new[idxs], box[idxs,4:5]], axis=-1)
  return img, box

def transform_hsv(img, h=0.015, s=0.7, v=0.4):
  r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
  hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
  x = np.arange(0, 256, dtype=r.dtype)
  lut_hue = (x * r[0] % 180).astype(np.uint8)
  lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
  lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
  img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
  return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

def transform_resize_and_pad(img, target_shape, pad_value=(114,114,114)):
  shape = img.shape[:2]
  r = min(target_shape[0]/shape[0], target_shape[1]/shape[1])
  unpad_shape = int(round(shape[0]*r)), int(round(shape[1]*r))
  if shape != unpad_shape:
    img = cv2.resize(img, unpad_shape[::-1], interpolation=cv2.INTER_CUBIC)
  tshape = target_shape
  dw, dh = (tshape[1] - img.shape[1]) / 2, (tshape[0] - img.shape[0]) / 2
  top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
  left, right = int(round(dw-0.1)), int(round(dw+0.1))
  # img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=114)
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)  # add border
  return img, (top, left)

def show_box(img, box, draw_center_point=False, verbose=True, format='yolo', use_overlay=True, num_state=1, show_conf=False, save_path=None, fontsize=12):
  from katacr.utils.detection import plot_box_PIL, build_label2colors
  from katacr.constants.label_list import idx2unit
  from katacr.constants.state_list import idx2state
  img = img.copy()
  if isinstance(img, np.ndarray):
    if img.max() <= 1.0: img *= 255
    img = Image.fromarray(img.astype('uint8'))
  # label_idx, conf_idx = (-1, None) if box.shape[1] == 13 else (-1, 4)
  if use_overlay:
    overlay = Image.new('RGBA', img.size, (0,0,0,0))  # build a RGBA overlay
  if len(box):
    label2color = build_label2colors(box[:,-1])
  for b in box:
    states = b[5:5+num_state] if len(b) == 7 else b[4:4+num_state]
    conf = float(b[4])
    label = int(b[-1])
    text = idx2unit[label]
    for i, s in enumerate(states):
      if i == 0:
        text += idx2state[int(s)]
      elif int(s) != 0:
        text += ' ' + idx2state[int(i*10 + s)]
    if show_conf:
      text += ' ' + f'{conf:.3f}'
    plot_box = lambda x: plot_box_PIL(
        x, b[:4],
        text=text,
        box_color=label2color[label],
        format=format, draw_center_point=draw_center_point,
        fontsize=fontsize
      )
    if use_overlay: overlay = plot_box(overlay)
    else: img = plot_box(img)
  if use_overlay:
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
  if verbose:
    img.show()
  if save_path is not None:
    img.save(str(save_path))
  return img
