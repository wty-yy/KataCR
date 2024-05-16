from PIL import Image
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from katacr.utils.detection import plot_cells_PIL, plot_box_PIL, build_label2colors
from typing import Tuple, List, Sequence
import numpy as np
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.build_dataset.constant import path_logs
from katacr.constants.label_list import unit2idx, idx2unit
from katacr.constants.state_list import state2idx, idx2state
from generation_config import (
  map_fly, map_ground, level2units, unit2level, grid_size, background_size, tower_unit_list, spell_unit_list,
  drop_units, xyxy_grids, towers_bottom_center_grid_position, drop_fliplr, 
  color2alpha, color2bright, color2RGB, aug2prob, aug2unit, alpha_transparency, background_augment,  # augmentation
  component_prob, component2unit, component_cfg, important_components, option_components, bar_xy_range,  # component configs
  item_cfg, drop_box, background_item_list,  # background item
  unit_scale, unit_stretch,  # affine transformation
  tower_intersect_ratio_thre, bar_intersect_ratio_thre, tower_generation_ratio, king_tower_generation_ratio
)
import random, glob

# background_size = (568, 896), cell_size = (30.9, 25)
cell_size = np.array([(xyxy_grids[3] - xyxy_grids[1]) / grid_size[1], (xyxy_grids[2] - xyxy_grids[0]) / grid_size[0]])[::-1]  # cell pixel: (w, h)

def cell2pixel(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

def pixel2cell(xy):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return ((xy - xyxy_grids[:2]) / cell_size).astype(np.float32)

def show_point(img: Image, xy_cell: tuple):
  xy = cell2pixel(xy_cell)
  img = plot_box_PIL(img, (xy[0],xy[1],5,5), draw_center_point=True)
  return img

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

class Unit:
  def __init__(
      self, img: np.ndarray,
      xy_bottom_center: tuple | np.ndarray,
      level: int,
      background_size: Tuple[int],
      name: str | None = None,
      cls: str | int | None = None,
      states: list | np.ndarray = None,
      fliplr: float = 0.5,
      augment: bool = True,
      drop: bool = False,
    ):
    """
    Args:
      img (np.ndarray): The segment unit image.
      xy_bottom_center (tuple | np.ndarray): The the bottom center position of the image relative to the cell grid.
      level (int): The level divide into 4:
        level 0: The ground units, and some weapons,
        level 1: The towers,
        level 2: The flying units,
        level 3: bar, clock, emoji, text.
      background_size (Tuple[int]): The image size of the background.
      name (str | None): The label name of the unit.
      cls (str | int | None): The class of the unit.
      states (str | np.ndarray): The states of the unit. (Side)
      background (bool): The image is the background.
      fliplr (float): The probability of flip the image left and right.
      augment (bool): If taggled, augment the image in probability.
      drop (bool): If taggled, the bounding box of the unit will be ignored.
    Variables:
      cls_name (str): The unit name.
      cls (int): The unit index.
      states (ndarray): The belonging of the unit.
      level (int): The layer level where the unit is on.
      xy_cell (tuple): The bottom center of the unit, relative to arena grids.np.
      xyxy (ndarray): The unit xyxy position relative to background.
      augment (bool): If taggled, the augment of color mask will be used.
      img (ndarray): The image of the unit, shape=(H,W,3)
      mask (ndarray): The available range of image, shape=(H,W), dtype=bool.
      components (List): The relative component units.
    """
    self.drop = drop
    self.components = []  # This will add by generator._add_components(unit)
    if name is not None or (cls is not None and states is not None):
      if name is not None:
        cls, *states = name.split('_')
      self.cls_name = cls if isinstance(cls, str) else idx2unit[cls]
      if isinstance(cls, str):
        if cls in drop_box or drop: self.cls = -1  # background items
        else: self.cls = unit2idx[cls]
      else: self.cls = cls
      if isinstance(states, np.ndarray):
        self.states = states
      elif self.cls_name not in drop_box:
        self.states = np.array((int(states[0]),), np.int32)
      else:
        self.states = np.array([0], np.int32)
        # self.states = np.zeros(7, np.int32)
        # for s in states:
        #   c, i = state2idx[s]
        #   self.states[c] = i
    else:
      raise "Error: You must give the label of the unit (when not background)."
    self.level = level

    if random.uniform(0, 1) < fliplr and self.cls_name not in drop_fliplr:
      img = np.fliplr(img)

    _sample_range = lambda l, r: random.random() * (r - l) + l
    if augment:  # scale and stretch
      if self.cls_name in unit_scale:
        r, prob = unit_scale[self.cls_name]
        if random.random() < prob:
          scale = _sample_range(*r)
          size = (np.array((img.shape[1], img.shape[0])) * scale).astype(np.int32)
          img = np.array(Image.fromarray(img).resize(size))
      if self.cls_name in unit_stretch:
        r, prob = unit_stretch[self.cls_name]
        if random.random() < prob:
          stretch = _sample_range(*r)
          size = (np.array((img.shape[1]*stretch, img.shape[0]))).astype(np.int32)
          img = np.array(Image.fromarray(img).resize(size))

    self.xy_cell = xy_bottom_center
    h, w = img.shape[:2]
    xy = cell2pixel(self.xy_cell)
    # Note that xyxy is (x0,y0,x1+1,y1+1)
    self.xyxy = np.array((xy[0]-w//2, xy[1]-h, xy[0]+(w+1)//2, xy[1]), np.float32)  # xyxy relative to background
    if self.cls_name in ['text', 'bar', 'circle'] + spell_unit_list:  # if text or spell units, clip the out range
      self.xyxy = np.array((
        max(self.xyxy[0], 0),
        max(self.xyxy[1], 0),
        min(self.xyxy[2], background_size[0]),
        min(self.xyxy[3], background_size[1])
      ), np.int32)
      # clip the image
      xyxy_relative = (self.xyxy - np.array([xy[0] - w // 2, xy[1] - h] * 2)).astype(np.int32)
      self.xyxy_relative = xyxy_relative
      img = img[xyxy_relative[1]:xyxy_relative[3], xyxy_relative[0]:xyxy_relative[2], :]
    else:  # not text, transpose into image range
      dxy = [0, 0]
      for i in range(2):
        if self.xyxy[i] < 0: dxy[i] = -self.xyxy[i]
        elif self.xyxy[2+i] > background_size[i]: dxy[i] = background_size[i] - self.xyxy[2+i]
      self.xyxy = np.array((
        self.xyxy[0] + dxy[0],
        self.xyxy[1] + dxy[1],
        self.xyxy[2] + dxy[0],
        self.xyxy[3] + dxy[1],
      ), np.int32)
    
    self.augment = None
    if augment:
      aug2prob_cp = aug2prob.copy()
      if self.cls_name == 'royal-ghost':
        aug2prob_cp['trans'] = 0.5
      if self.cls_name == 'archer-queen':
        aug2prob_cp['trans'] = 0.3
      p = random.random()
      for key, val in aug2prob_cp.items():
        if self.cls_name not in aug2unit[key]: continue
        if p < val:
          self.augment = key
          break
        p -= val
    
    size = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
    # Residue size ratio < 0.3 or width, hight < 6 pixel, then drop this unit
    if size / (h * w) < 0.3 or self.xyxy[2] - self.xyxy[0] < 6 or self.xyxy[3] - self.xyxy[1] < 6:
      self.xyxy = np.zeros(4)

    if img.shape[-1] == 4:
      self.mask = img[...,3] > 0
      self.img = img[...,:3]
    else:
      self.mask = img.sum(-1) > 0
      self.img = img

  def get_name(self, show_state=True):
    name = self.cls_name
    if not show_state: return name
    for i, s in enumerate(self.states):
      if i == 0:
        name += idx2state[s]
      elif s != 0:
        name += '_' + idx2state[i*10+s]
    return name
  
  def draw(self, img: np.ndarray, inplace=True):
    xyxy = self.xyxy
    mask = self.mask_visiable
    if not inplace: img = img.copy()
    if self.augment is not None:
      if self.augment == 'trans':  # Transparency augmentation
        subimg = img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:]
        alpha = (mask * alpha_transparency).astype(np.uint8)
        upimg = Image.fromarray(np.concatenate([self.img, alpha[...,None]], -1))
        upimg = np.array(Image.alpha_composite(Image.fromarray(subimg).convert('RGBA'), upimg).convert('RGB'))
        subimg[mask] = upimg[mask]
      else:  # color filter, self.augment in ['red', 'blue', 'golden', 'white']
        color = self.augment
        bright = random.randint(*color2bright[color])
        upimg = add_filter(self.img, color, color2alpha[color], bright, replace=False)
        img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:][mask] = upimg[mask]
    else:
      img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:][mask] = self.img[mask]
    return img

  def update_xyxy(self, mask: np.ndarray):
    self.xyxy_visiable = np.zeros_like(self.xyxy)
    if self.xyxy.sum() == 0: return
    mask = mask[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2]]
    self.mask_visiable = (self.mask ^ mask) & self.mask
    xyxy_ = np.array([  # Minimal uncovered image
      np.argwhere(self.mask_visiable.any(0))[[0,-1],0],
      np.argwhere(self.mask_visiable.any(1))[[0,-1],0],
    ]).T.reshape(-1)
    w, h = xyxy_[2] - xyxy_[0], xyxy_[3] - xyxy_[1]
    if w < 6 or h < 6: return
    self.xyxy_visiable = np.array([
      self.xyxy[0] + xyxy_[0],
      self.xyxy[1] + xyxy_[1],
      self.xyxy[0] + xyxy_[0] + w,
      self.xyxy[1] + xyxy_[1] + h,
    ])

  def draw_mask(self, mask: np.ndarray, inplace=True):
    if not inplace: mask = mask.copy()
    mask[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2]][self.mask] = 1
    return mask

  def show_box(self, img: Image, cls2color: dict | None = None):
    if self.cls == -1: return img
    color = cls2color[self.cls] if cls2color is not None else 'red'
    # return plot_box_PIL(img, self.xyxy_visiable, text=self.get_name(), format='voc', box_color=color)  # xyxy visiable is bad
    return plot_box_PIL(img, self.xyxy, text=self.get_name(), format='voc', box_color=color)

class Generator:
  towers_bc_pos: dict = towers_bottom_center_grid_position  # with keys: ['king0', 'king1', 'queen0_0', 'queen0_1', 'queen1_0', 'queen1_1']

  def __init__(
      self, background_index: int | None = None,
      unit_list: Tuple[Unit,...] = None,
      seed: int | None = None,
      intersect_ratio_thre: float = 0.5,
      map_update: dict = {'mode': 'naive','size': 5},
      augment: bool = True,
      dynamic_unit: bool = True,
      avail_names: Sequence[str] = None,
      noise_unit_ratio: float = 0.0,
    ):
    """
    Args:
      background_index: Use image file name in `dataset/images/segment/backgrounds/background{index}.jpg` as current background.
      unit_list: The list of units will be generated in arean.
      seed: The random seed.
      intersect_ratio_thre: The threshold to filte overlapping units.
      map_update_size: The changing size of dynamic generation distribution (SxS).
      augment: If taggled, the mask augmentation will be used.
      dynamic_unit: If taggled, the frequency of each unit will tend to average.
      avail_names: Specify the generation classes.
      noise_unit_ratio: The ratio of inavailable unit (noise unit) in whole units.
    Variables:
      map_cfg (dict):
        'ground': The 0/1 ground unit map in `katacr/build_dataset/generation_config.py`.
        'fly': The 0/1 fly unit map in `katacr/build_dataset/generation_config.py`.
        'update_size': The size of round squra.
    """
    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)
    self.background_index = background_index
    self.augment = augment
    self.dynamic_unit = dynamic_unit
    self.noise_unit_ratio = noise_unit_ratio

    self.path_manager = PathManager()
    self.path_segment = self.path_manager.path / "images/segment"
    self.build_background()
    self.unit_list = [] if unit_list is None else unit_list
    self.intersect_ratio_thre = intersect_ratio_thre
    self.map_cfg = {
      'ground': np.array(map_ground, np.float32),
      'fly': np.array(map_fly, np.float32),
    }
    self.map_cfg.update(map_update)
    self.moveable_unit_paths, self.moveable_unit2idx, self.idx2moveable_unit = [], {}, {}
    self.noise_unit_paths, self.noise_unit2idx, self.idx2noise_unit = [], {}, {}
    self.avail_names = avail_names
    for p in sorted(self.path_segment.glob('*')):
      if p.name in ['backgrounds'] + tower_unit_list + drop_units:
        continue
      for p_img in p.glob('*.png'):
        assert p_img.name.split('_')[0] == p.name, f"ERROR segment image name: {p_img}, make sure the prefix name of segment image is same as its parent directory name."
      for bel in range(2):
        p_name = f"{p.name}_{bel}"
        p_bel = str(p / (p_name + '*.png'))
        if len(glob.glob(p_bel)):
          if avail_names is not None and p.name not in avail_names:  # noise unit
            paths, u2i, i2u = self.noise_unit_paths, self.noise_unit2idx, self.idx2noise_unit
          else:
            paths, u2i, i2u = self.moveable_unit_paths, self.moveable_unit2idx, self.idx2moveable_unit
          i = len(self.noise_unit_paths)
          u2i[p_name] = i
          i2u[i] = p_name
          paths.append(p_bel)
    self.moveable_unit_frequency = np.zeros(len(self.moveable_unit_paths), np.int32)
    self.noise_unit_frequency = np.zeros(len(self.noise_unit_paths), np.int32)
  
  def build_background(self):
    background_index = self.background_index
    if background_index is None:
      n = len(self.path_manager.search(subset='images', part='segment', name='backgrounds', regex="background\d+.jpg"))
      background_index = random.randint(1, n)
    self.background = np.array(Image.open(self.path_manager.path / f"images/segment/backgrounds/background{background_index:02}.jpg"))
    self.background_size = self.background.shape[:2][::-1]
    assert self.background_size[0] == background_size[0] and self.background_size[1] == background_size[1]

    if self.augment:
      p = random.random()
      if p < background_augment['prob']:
        add_filter(self.background, 'red', alpha=100, xyxy=background_augment['xyxy'])
        red_bound = Image.open(str(self.path_segment / 'backgrounds/red_bound.png'))
        self.background = np.array(Image.alpha_composite(Image.fromarray(self.background).convert('RGBA'), red_bound).convert('RGB'))
  
  @staticmethod
  def _max_intersect_ratio(xyxy: tuple | np.ndarray, box: List[np.ndarray] | np.ndarray):
    if not isinstance(box, np.ndarray): box = np.array(box)
    if not isinstance(xyxy, np.ndarray): xyxy = np.array(xyxy)
    xyxy = xyxy.reshape(1, 4)
    box = box.reshape(-1, 4)
    min1, min2 = xyxy[:,:2], box[:,:2]
    max1, max2 = xyxy[:,2:], box[:,2:]
    inter_h = (np.minimum(max1[...,0],max2[...,0]) - np.maximum(min1[...,0],min2[...,0])).clip(0.0)
    inter_w = (np.minimum(max1[...,1],max2[...,1]) - np.maximum(min1[...,1],min2[...,1])).clip(0.0)
    inter_size = inter_h * inter_w
    xyxy = xyxy[0]
    xyxy_size = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
    if xyxy_size == 0: return 1.0
    return inter_size.max() / xyxy_size
  
  @staticmethod
  def _intersect_ratio_with_mask(unit: Unit, mask: np.ndarray):
    if (unit.xyxy == 0).all(): return 1.0
    if unit.mask.sum() == 0: return 0.0
    inter = mask[unit.xyxy[1]:unit.xyxy[3],unit.xyxy[0]:unit.xyxy[2]][unit.mask].sum()
    return inter / unit.mask.sum()
  
  def xyxy2cxcywh(self, xyxy, img_shape, relative=True):
    if xyxy.ndim == 1: xyxy = xyxy.reshape(1, 4)
    bx, by = np.round(xyxy[:,[0,2]].sum(-1)/2), np.round(xyxy[:,[1,3]].sum(-1)/2)
    bw, bh = np.abs(xyxy[:,2]-xyxy[:,0]), np.abs(xyxy[:,3]-xyxy[:,1])
    w, h = img_shape[:2][::-1]
    bw = np.minimum(bw, np.minimum(bx, w - bx) * 2)
    bh = np.minimum(bh, np.minimum(by, h - by) * 2)
    if relative:
      bx, bw = bx / w, bw / w
      by, bh = by / h, bh / h
    return np.stack([bx, by, bw, bh], -1)

  def xyxy2xywh(self, xyxy, img_shape, relative=True):
    if xyxy.ndim == 1: xyxy = xyxy.reshape(1, 4)
    x, y = xyxy[:, 0], xyxy[:, 1]
    w, h = xyxy[:, 2] - x, xyxy[:, 3] - y
    if relative:
      ws, hs = img_shape[:2][::-1]
      x, w = x / ws, w / ws
      y, h = y / hs, h / hs
    return np.stack([x, y, w, h], -1)
  
  def resize_and_pad(self, img, box, img_size):
    import cv2
    # resize image and padding to img_size (896, 568) -> (896, 576)
    shape = img.shape[:2]
    target_shape =  img_size[::-1]
    r = min(target_shape[0]/shape[0], target_shape[1]/shape[1])
    unpad_shape = int(round(shape[0]*r)), int(round(shape[1]*r))
    if shape != unpad_shape:
      img = cv2.resize(img, unpad_shape[::-1], interpolation=cv2.INTER_CUBIC)
    dw, dh = target_shape[1] - unpad_shape[1], target_shape[0] - unpad_shape[0]  # wh padding
    dw, dh = dw / 2, dh / 2  # put image in padding center
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    box[:, :4] *= r
    box[:, [0, 2]] += left
    box[:, [1, 3]] += top
    return img, box
  
  def build(self, save_path="", verbose=False, show_box=False, box_format='cxcywh', img_size=None):
    img = self.background.copy()
    cls, box = set(), []
    self.unit_list = sorted(self.unit_list, key=lambda x: (x.level, x.xy_cell[1]))
    while True:
      mask = np.zeros(img.shape[:2], dtype=np.bool_)  # union of images
      unit_avail: List[Unit] = []  # available units
      tmp_unit_list = self.unit_list.copy()
      for u in self.unit_list[::-1]:  # reverse order for NMS
        ratio = self._intersect_ratio_with_mask(u, mask)
        keep_unit = True
        if u.cls_name in tower_unit_list:
          if ratio > tower_intersect_ratio_thre: keep_unit = False
        elif u.cls_name == 'bar':
          if ratio > bar_intersect_ratio_thre: keep_unit = False
        elif ratio > self.intersect_ratio_thre: keep_unit = False
        if not keep_unit:
          if u in tmp_unit_list:
            tmp_unit_list.remove(u)
          for cu in u.components:
            if cu in tmp_unit_list:
              tmp_unit_list.remove(cu)
        else:
          u.update_xyxy(mask)
          u.draw_mask(mask)
          unit_avail.append(u)
      self.unit_list = tmp_unit_list
      if len(self.unit_list) == len(unit_avail): break
    for u in unit_avail[::-1]:  # increase order for drawing
      u.draw(img)
      cls.add(u.cls)
      if u.cls_name in self.moveable_unit2idx:
        self.moveable_unit_frequency[self.moveable_unit2idx[u.cls_name+'_'+str(u.states[0])]] += 1
      if u.cls_name in self.noise_unit2idx:
        self.noise_unit_frequency[self.noise_unit2idx[u.cls_name+'_'+str(u.states[0])]] += 1
      if u.cls != -1:
        # box.append((*u.xyxy_visiable, *u.states, u.cls))  # xyxy visiable is bad
        box.append((*u.xyxy, *u.states, u.cls))
    box = np.array(box, np.float32) if len(box) else np.empty((0, 6), np.float32)
    if img_size is not None:
      img, box = self.resize_and_pad(img, box, img_size)
    if box_format == 'cxcywh':
      box[:,:4] = self.xyxy2cxcywh(box[:,:4], img.shape)
    elif box_format == 'xywh':
      box[:,:4] = self.xyxy2xywh(box[:,:4], img.shape)
    else:
      raise RuntimeError(f"Don't know {box_format=}")
    origin_img = img
    img = Image.fromarray(img)
    if show_box:
      cls2color = build_label2colors(list(cls))
      for u in unit_avail:
        img = u.show_box(img, cls2color)
    if len(save_path): img.save(save_path)
    if verbose: img.show()
    pil_img = img
    return origin_img, box, pil_img
  
  def join(self, unit: Unit):
    assert isinstance(unit, Unit), "The join element must be the instance of Unit"
    self.unit_list.append(unit)
  
  def _build_unit_from_path(
      self, path: str | Path,
      xy: tuple,
      level: int,
      max_width: float | tuple = None,  # relative to cell
      xy_format: str = 'bottom_center',  # decide the format of 'xy'
      drop: bool = False,
      join: bool = True,
    ):
    """
    Args:
      path (str | Path): The image path of the unit.
      xy (tuple): The xy position of the unit which satisfied `xy_format`.
      level (int): The graphic layer level of the unit.
      max_width (float | tuple): The max width of the unit which relative to cell.
      xy_format (str): The xy position format which has these options: 
        ['bottom_center', 'center', 'left_center', 'right_center',
         'left_bottom', 'right_bottom', 'top_center']
      drop (bool): If taggled, the unit's box will be drop, which is used to the noise units.
      join (bool): If taggled, the unit will be joined to `self.unit_list`.
    """
    path = Path(path)
    avail_format = ['bottom_center', 'center', 'left_center', 'right_center', 'left_bottom', 'right_bottom', 'top_center']
    assert xy_format in avail_format, f"Need 'xy_format' in {avail_format}"
    img = Image.open(path)
    if max_width is not None:
      if isinstance(max_width, tuple): max_width = self._sample_range(*max_width)
      max_width = round(max_width * cell_size[0])  # convert to pixel
      if img.size[0] > max_width:
        ratio = max_width / img.size[0]
        img = img.resize((round(img.size[i]*ratio) for i in range(2)))
    img = np.array(img)
    name = path.name.rsplit('.',1)[0]
    name = name.rsplit('_',1)[0]
    xy_bottom_center = list(xy)
    if xy_format in ['center', 'left_center', 'right_center']:
      xy_bottom_center[1] += img.shape[0] / 2 / cell_size[1]
    if xy_format in ['top_center']:
      xy_bottom_center[1] += img.shape[0] / cell_size[1]
    if 'left' in xy_format:
      xy_bottom_center[0] += img.shape[1] / 2 / cell_size[0]
    if 'right' in xy_format:
      xy_bottom_center[0] -= img.shape[1] / 2 / cell_size[0]
    if (self.avail_names is not None) and (name.split('_')[0] not in self.avail_names): drop = True
    unit = Unit(img=img, xy_bottom_center=xy_bottom_center, level=level, background_size=self.background_size, name=name, augment=self.augment, drop=drop)
    if join: self.unit_list.append(unit)
    return unit
  
  @staticmethod
  def _sample_elem(x, k=1, get_elem=True):
    ret = random.sample(x, k=k)
    if k == 1 and get_elem: return ret[0]
    return ret

  @staticmethod
  def _sample_prob(a: np.ndarray, size: int = 1, replace: bool = False) -> np.ndarray:
    """
    Sample the `size` indexs of the distribution matrix `a`.

    Args:
      a (np.ndarray): The distribution matrix `a`.
      size (int): The sample size of indexs.
      replace (bool): If true, allow to use repeatable indexs.
    """
    r_idx = np.random.choice(np.arange((a!=0).sum()), size, replace=replace, p=a[a!=0] / a[a!=0].sum())
    return np.argwhere(a)[r_idx]
  
  @staticmethod
  def _sample_range(l, r):
    return random.random() * (r - l) + l
  
  def add_tower(self, king=True, queen=True):
    if king:
      for i in range(2):  # is enermy?
        if random.random() < king_tower_generation_ratio:
          path = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='king-tower', regex=f'king-tower_{i}'))
        else:
          path = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='background-items', regex='^king-tower-ruin_\d+.png'))
        unit = self._build_unit_from_path(path, self.towers_bc_pos[f'king{i}'], 1)
        self._add_component(unit)
      # king0 = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='king-tower', regex='king-tower_0'))
      # unit = self._build_unit_from_path(king0, self.towers_bc_pos['king0'], 1)
      # self._add_component(unit)
      # king1 = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='king-tower', regex='king-tower_1'))
      # unit = self._build_unit_from_path(king1, self.towers_bc_pos['king1'], 1)
      # self._add_component(unit)
    if queen:
      for i in range(2):  # is enermy?
        for j in range(2):  # left or right
          p = random.random()
          if p > sum(list(tower_generation_ratio.values())):  # generating tower ruin in background_itmes/ruin{i}.png
            path = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='background-items', regex='^ruin_\d+.png'))
            # continue  # TODO: Add tower ruin
          else:
            for name, prob in tower_generation_ratio.items():
              # if p <= prob or i == 0: break  # our tower must use queen tower (we don't have)
              if p <= prob: break
              p -= prob
            path = self._sample_elem(self.path_manager.search(subset='images', part='segment', name=name, regex=f"{name}_{i}"))
          unit = self._build_unit_from_path(path, self.towers_bc_pos[f'queen{i}_{j}'], 1)
          unit = self._add_component(unit)
          
  @staticmethod
  def _update_map(
    a: np.ndarray,
    xy: np.ndarray | Sequence,
    size: int = 3,
    mode: str = 'dynamic',
    replace: bool = True
  ) -> np.ndarray:
    """
    Dynamic update mode:
      Update the distribution map after use the `xy` position, modify policy is:
    confirm the center point `xy`, find the `(size, size)` area around the center,
    let `a[*xy] <- a[*xy] / 2`, around values add `a[*xy] / 2 / num_round`,
    for example (left matrix is `a`, let `xy=(1,2), size=3`):
    ```
    [[1. 1. 0.]     [[1.    1.125 0.   ]
      [0. 1. 1.]  ->  [0.    1.125 0.5  ]
      [0. 1. 1.]]     [0.    1.125 1.125]]
    ```

    Naive update mode:
      Just let `a[*xy] <- 0`.

    Args:
      a (np.ndarray): The distribution map (not necessary sum to 1).
      xy (np.ndarray | Sequence): The point to place the unit.
      size (int): The size of the squre width.
      mode (str): The mode of update map, dynamic or naive.
      replace (bool): Replace the array `a` directly.
    Return:
      a (np.ndarray): The distribution map after updating.
    """
    assert mode in ['dynamic', 'naive'], f"map update mode must in ['dynamic', 'naive']"
    if not replace: a = a.copy()
    if not isinstance(xy, np.ndarray):
      xy = np.array(xy)
    assert a[xy[0],xy[1]] != 0 and xy.size == 2

    if mode == 'dynamic':
      d = np.stack(np.meshgrid(np.arange(size), np.arange(size)), -1) - size // 2  # delta
      dxy = (d + xy.reshape(1,1,2)).reshape(-1, 2).T  # xy around indexs with sizexsize
      for i in range(2):  # clip the indexs which are out of range
        dxy[i] = dxy[i].clip(0, a.shape[i]-1)
      dx, dy = np.unique(dxy, axis=1)  # unique the same indexs
      nonzero = (a[dx, dy] != 0) ^ ((dx == xy[0]) & (dy == xy[1]))  # get nonzeros mask in around and remove center index
      n = nonzero.sum()  # total number of nonzeros (without center point)
      if n == 0: return
      dx, dy = dx[nonzero], dy[nonzero]

      c = a[xy[0],xy[1]]  # center value
      a[xy[0],xy[1]] = c / 2  # update center point
      a[dx, dy] += c / 2 / n  # update around point
    elif mode == 'naive':
      a[xy[0],xy[1]] = 0
    return a

  def _sample_from_map(self, level: int, noisy: bool = True, replace_map: bool = True):
    """
    Sample one position with non-zero element from the 2d distribution map array,
    and update the map.
    
    Args:
      level: 1 is ground, [2,3] is fly and others. (see generation_config.py)
      noisy (bool): If true, add Gaussian noisy (mu=0, sigma=0.4) to the position.
      replace_map (bool): If true, update the `map` inplace.
    """
    map = self.map_cfg['ground'] if level == 1 else self.map_cfg['fly']
    if not replace_map: map = map.copy()
    assert isinstance(map, np.ndarray) and map.dtype == np.float32, "The distribution map must be ndarray and float32"
    xy = self._sample_prob(map)[0]  # array index
    self._update_map(map, xy, size=self.map_cfg['size'], mode=self.map_cfg['mode'])
    xy = xy.astype(np.float32)[::-1] + 0.5  # array idx -> center of the cell
    if noisy:
      xy += np.clip(np.random.randn(2) * 0.2, -0.5, 0.5)
      xy = np.array([np.clip(xy[0], 0, grid_size[0]), np.clip(xy[1], 0, grid_size[1])])
    return xy

  def _sample_from_center(self, center, dx_range, dy_range):
    dx = self._sample_range(*dx_range)
    dy = self._sample_range(*dy_range)
    if not isinstance(center, np.ndarray): center = np.array(center, np.float32)
    center += np.array([dx, dy])
    return center
  
  def _add_component(self, unit: Unit):
    components = [key for key, (units, prob) in component2unit.items() if unit.cls_name in units]
    if len(components) == 0: return
    cs = []  # seleted components
    # First, check important component probability
    for c, prob in important_components:
      if c == ('bar', 'bar-level'):
        prob = 1.0 if unit.states[0] == 1 else 0.25
      if c in components and c not in cs and random.random() < prob:
        components.remove(c)
        cs.append(c)
    for oc, prob in option_components:
      if oc in components:
        rand = random.random()
        for c, p in zip(oc, prob):
          if rand > p: rand -= p
          else: break
        components.remove(oc)
        cs.append(c)
    # Second, check normal component probability
    prob = component_prob[unit.cls_name]
    if random.random() < prob:
      # Third, check single component probability
      for c in components:
        if random.random() < component2unit[c][1]:
          cs.append(c)
    if not len(cs): return  # no selected components
    for c in cs:
      if isinstance(c, tuple):
        c = self._sample_elem(c)
        # if random.random() < 0.5: c = c[0]  # 0.3 prob for 'bar'
        # else: c = c[1]  # 0.7 prob for 'bar-level'
        # DEBUG:
        # c = 'bar'
      if c in component_cfg: cfg = component_cfg[c]
      else: cfg = component_cfg[c+str(unit.states[0])]
      center, dx_range, dy_range, max_width, xy_format = cfg
      if center == 'bottom_center': center = unit.xy_cell
      elif center == 'top_center': center = (unit.xy_cell[0], (unit.xyxy[1]-xyxy_grids[1])/cell_size[1])  # xyxy_grids[1] is the top of whold grid in image
      elif center == 'center': center = (unit.xy_cell[0], ((unit.xyxy[1]+unit.xyxy[3])/2-xyxy_grids[1])/cell_size[1])
      xy = self._sample_from_center(center, dx_range, dy_range)
      path = self.path_segment / c
      if 'bar' in c and c != 'dagger-duchess-tower-bar':  # determine the side 0/1
        paths = sorted(path.glob(f"{c}_{unit.states[0]}*"))
      elif c in background_item_list:
        paths = sorted((self.path_segment / 'background-items').glob(c + '*'))
      else:
        paths = sorted(path.glob('*'))
      if len(paths) == 0: return
      level = unit2level[c]
      if c == 'bar':  # update xy, xy is junction of bar-level and bar
        xy[0] += self._sample_range(*bar_xy_range)
        xy_ = xy.copy()
        xy_[0] += 0.08  # 0.08 * 30.8 pixel = 2.464 pixel
        paths_bar_level = sorted(self.path_manager.path.joinpath("images/segment/bar-level").glob(f"bar-level_{unit.states[0]}*"))
        bar_level = self._build_unit_from_path(self._sample_elem(paths_bar_level), xy_, level, None, 'right_center')
        unit.components.append(bar_level)
      cu = self._build_unit_from_path(self._sample_elem(paths), xy, level, max_width, xy_format)
      unit.components.append(cu)
      if c == 'dagger-duchess-tower-bar':
        junction = pixel2cell((cu.xyxy[0], (cu.xyxy[1] + cu.xyxy[3]) / 2))
        junction[1] += 0.06
        bar_icon = self._build_unit_from_path(self.path_segment/"background-items/dagger-duchess-tower-icon.png", junction, level, None, 'right_center')
        unit.components.append(bar_icon)
  
  def _add_background_item(self):
    """
    Add background items in `background_item_list`, `big-text`, `small-text`, `emote`.
    Look at `item_cfg.keys()`.
    """
    # (prob, [center, dx_range, dy_range, width_range, max_num]*n)
    for name, (prob, cfgs) in item_cfg.items():
      if random.random() > prob: continue
      pure_name = file_name = name              # For example:
      if name[-1] in ['0', '1']:                # name = scoreboard0
        pure_name = name[:-1]                   # pure_name = scoreboard
        file_name = pure_name + '_' + name[-1]  # file_name = scoreboard_0
      for cfg in cfgs:
        if pure_name not in background_item_list:
          paths = sorted((self.path_segment / pure_name).glob('*'))
        else:
          paths = sorted((self.path_segment / 'background-items').glob(file_name+'*'))
        level = unit2level[pure_name]  # [0: background_items, 3: big-text, emote]
        center, dx_range, dy_range, w_range, maxn = cfg
        n = self._sample_elem(range(maxn)) + 1
        # if maxn == 1:
        #   n = self._sample_elem(range(2))
        # else: n = self._sample_elem(range(maxn)) + 1
        for _ in range(n):
          xy = self._sample_from_center(center, dx_range, dy_range)
          path = self._sample_elem(paths)
          self._build_unit_from_path(path, xy, level, w_range)
  
  def add_unit(self, n=1):
    """
    Add unit in [ground, flying, others] randomly.
    Unit list looks at `katacr/constants/label_list.py`
    """
    # self._add_background_item()
    def get_freq(freq):
      if self.dynamic_unit and len(freq):
        return 1 / (freq - freq.min() + 1)
      else:
        return np.ones_like(freq)
    noise_freq = get_freq(self.noise_unit_frequency)
    moveable_freq = get_freq(self.moveable_unit_frequency)
    def add_compoents(freq, drop, paths):
      if len(freq) == 0: return
      ratio = self.noise_unit_ratio if drop else (1 - self.noise_unit_ratio)
      idxs = self._sample_prob(freq, size=int(n*ratio), replace=True).reshape(-1)
      for i in idxs:
        p = paths[i]
        level = unit2level[Path(p).name.split('_')[0]]  # [1, 2, 3]
        xy = self._sample_from_map(level)
        unit = self._build_unit_from_path(self._sample_elem(sorted(glob.glob(p))), xy, level, drop=drop)
        self._add_component(unit)
    add_compoents(noise_freq, True, self.noise_unit_paths)
    add_compoents(moveable_freq, False, self.moveable_unit_paths)
  
  def reset(self):
    self.build_background()
    self.unit_list = []
    self.map_cfg.update({
      'ground': np.array(map_ground, np.float32),
      'fly': np.array(map_fly, np.float32),
    })

if __name__ == '__main__':
  # generator = Generator(seed=42, background_index=25, intersect_ratio_thre=0.5, augment=True, map_update={'mode': 'naive', 'size': 5}, avail_names=None)
  generator = Generator(seed=1, background_index=25, intersect_ratio_thre=0.5, augment=True, map_update={'mode': 'dynamic', 'size': 5}, avail_names=None)
  # generator = Generator(seed=42, intersect_ratio_thre=0.5, augment=True, map_update={'mode': 'dynamic', 'size': 5}, noise_unit_ratio=1/4, avail_names=['king-tower', 'queen-tower', 'cannoneer-tower', 'dagger-duchess-tower', 'dagger-duchess-tower-bar', 'tower-bar', 'king-tower-bar', 'bar', 'bar-level', 'clock', 'emote', 'elixir', 'ice-spirit-evolution-symbol', 'evolution-symbol', 'bat', 'elixir-golem-small', 'fire-spirit', 'skeleton', 'lava-pup', 'skeleton-evolution', 'heal-spirit', 'ice-spirit', 'phoenix-egg', 'bat-evolution', 'minion', 'goblin', 'archer', 'spear-goblin', 'bomber', 'electro-spirit', 'royal-hog', 'rascal-girl', 'ice-spirit-evolution', 'hog', 'dirt', 'mini-pekka', 'wizard', 'barbarian', 'zappy', 'little-prince', 'firecracker', 'valkyrie', 'bandit', 'wall-breaker', 'musketeer', 'princess', 'barbarian-evolution', 'elite-barbarian', 'guard', 'knight-evolution', 'archer-evolution', 'bomber-evolution', 'goblin-brawler', 'bomb', 'goblin-ball', 'axe', 'electro-wizard', 'mother-witch', 'elixir-golem-mid', 'tesla', 'knight', 'royal-recruit', 'ice-wizard', 'valkyrie-evolution', 'dart-goblin', 'mortar', 'the-log', 'firecracker-evolution', 'lumberjack', 'royal-ghost', 'miner', 'night-witch', 'ram-rider', 'electro-dragon', 'hunter', 'mortar-evolution', 'executioner', 'mega-minion', 'golemite', 'witch', 'barbarian-barrel'])
  path_generation = path_logs / "generation"
  path_generation.mkdir(exist_ok=True)
  for i in range(1):
    # generator = Generator(background_index=None, seed=42+i, intersect_ratio_thre=0.9)
    # generator._add_background_item()
    # generator.add_tower()
    # generator.add_unit(n=15)
    x, box, _ = generator.build(verbose=False, show_box=True, save_path=str(path_generation / f"test{0+2*i}.jpg"))
    # for b in box:
    #   assert idx2unit[b[5]] != 'skeleton-king-skill'
    # x, box, _ = generator.build(verbose=True, show_box=False, save_path=str(path_generation / f"test{0+2*i}.jpg"))
    # print(generator.moveable_unit_frequency)
    # f = generator.moveable_unit_frequency
    # for i in range(len(f)):
    #   if f[i] > 0:
    #     print(f"{generator.idx2moveable_unit[i]}: {f[i]}")
    generator.build(verbose=False, show_box=False, save_path=str(path_generation / f"test{0+2*i+1}.jpg"))
    print('box num:', box.shape[0])
    # print(generator.map_cfg['ground'])
    generator.reset()
  # generator = Generator(background_index=None, seed=42, intersect_ratio_thre=0.6)
  # generator.add_tower()
  # generator.add_unit(n=80)
  # x, box = generator.build()
  # img = Image.fromarray(x)
  # from katacr.utils.detection.data import show_box
  # # box = np.roll(box, -1, -1)
  # img = show_box(img, box, verbose=False)
  # print(x.shape, box.shape)
  # img.show()
