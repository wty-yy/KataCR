from PIL import Image
from pathlib import Path
from katacr.utils.detection import plot_cells_PIL, plot_box_PIL, build_label2colors
from typing import Tuple, List, Sequence
import numpy as np
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.build_dataset.constant import path_logs
from katacr.constants.label_list import unit2idx, idx2unit
from katacr.constants.state_list import state2idx, idx2state
from katacr.build_dataset.generation_config import (
  map_fly, map_ground, level2units, unit2level, grid_size, background_size, tower_unit_list,
  drop_units, xyxy_grids, bottom_center_grid_position, drop_fliplr, 
  color2alpha, color2bright, color2RGB, aug2prob, aug2unit, alpha_transparency, background_augment,  # augmentation
  component_prob, component2unit, component_cfg, important_components,  # component configs
  item_cfg, drop_box, background_item_list  # background item
)
import random

cell_size = np.array([(xyxy_grids[3] - xyxy_grids[1]) / grid_size[1], (xyxy_grids[2] - xyxy_grids[0]) / grid_size[0]])[::-1]  # cell pixel: (w, h)

def cell2pixel(xy: tuple):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

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
      augment (bool): If true, augment the image in probability.
    """
    if name is not None or (cls is not None and states is not None):
      if name is not None:
        cls, *states = name.split('_')
      self.cls_name = cls if isinstance(cls, str) else idx2unit[cls]
      if isinstance(cls, str):
        if cls in drop_box: self.cls = -1
        else: self.cls = unit2idx[cls]
      else: self.cls = cls
      if isinstance(states, np.ndarray):
        self.states = states
      elif self.cls_name not in drop_box:
        self.states = np.array((int(states[0]),), np.int32)
        # self.states = np.zeros(7, np.int32)
        # for s in states:
        #   c, i = state2idx[s]
        #   self.states[c] = i
    else:
      raise "Error: You must give the label of the unit (when not background)."
    self.level = level

    self.xy_cell = xy_bottom_center
    h, w = img.shape[:2]
    xy = cell2pixel(self.xy_cell)
    self.xyxy = np.array((xy[0]-w//2, xy[1]-h, xy[0]+(w+1)//2, xy[1]), np.float32)  # xyxy relative to background
    if self.cls == unit2idx['text']:  # if text, clip the out range
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
    
    size = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
    # Residue size ratio < 0.3 or width, hight < 6 pixel, then drop this unit
    if size / (h * w) < 0.3 or self.xyxy[2] - self.xyxy[0] < 6 or self.xyxy[3] - self.xyxy[1] < 6:
      self.xyxy = np.zeros(4)

    if random.uniform(0, 1) < fliplr and self.cls_name not in drop_fliplr:
      img = np.fliplr(img)

    if img.shape[-1] == 4:
      self.mask = img[...,3] > 0
      self.img = img[...,:3]
    else:
      self.mask = img.sum(-1) > 0
      self.img = img
    
    self.augment = None
    if augment:
      p = random.random()
      for key, val in aug2prob.items():
        if self.cls_name not in aug2unit[key]: continue
        if p < val:
          self.augment = key
          break
        p -= val

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
    if not inplace: img = img.copy()
    if self.augment is not None:
      if self.augment == 'trans':  # Transparency augmentation
        subimg = img[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2],:]
        alpha = ((self.mask != 0) * alpha_transparency).astype(np.uint8)
        upimg = Image.fromarray(np.concatenate([self.img, alpha[...,None]], -1))
        upimg = np.array(Image.alpha_composite(Image.fromarray(subimg).convert('RGBA'), upimg).convert('RGB'))
        subimg[self.mask] = upimg[self.mask]
      else:  # color filter, self.augment in ['red', 'blue', 'golden', 'white']
        color = self.augment
        bright = random.randint(*color2bright[color])
        upimg = add_filter(self.img, color, color2alpha[color], bright, replace=False)
        img[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2],:][self.mask] = upimg[self.mask]
    else:
      img[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2],:][self.mask] = self.img[self.mask]
    return img
  
  def draw_mask(self, mask: np.ndarray, inplace=True):
    if not inplace: mask = mask.copy()
    mask[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2]][self.mask] = 1
    return mask

  def show_box(self, img: Image, cls2color: dict | None = None):
    if self.cls == -1: return img
    color = cls2color[self.cls] if cls2color is not None else 'red'
    return plot_box_PIL(img, self.xyxy, text=self.get_name(), format='voc', box_color=color)

class Generator:
  bc_pos: dict = bottom_center_grid_position  # with keys: ['king0', 'king1', 'queen0_0', 'queen0_1', 'queen1_0', 'queen1_1']

  def __init__(
      self, background_index: int | None = None,
      unit_list: Tuple[Unit,...] = None,
      seed: int | None = None,
      intersect_ratio_thre: float = 0.5,
      tower_intersect_ratio_thre: float = 0.8,
      map_update_size: int = 5,
      augment: bool = True
    ):
    """
    Args:
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

    self.path_manager = PathManager()
    self.path_segment = self.path_manager.path / "images/segment"
    self.build_background()
    self.unit_list = [] if unit_list is None else unit_list
    self.intersect_ratio_thre = intersect_ratio_thre
    self.tower_intersect_ratio_thre = tower_intersect_ratio_thre
    self.map_cfg = {
      'ground': np.array(map_ground, np.float32),
      'fly': np.array(map_fly, np.float32),
      'update_size': map_update_size
    }
  
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
        add_filter(self.background, 'red', alpha=80, xyxy=background_augment['xyxy'])
  
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
  
  def xyxy2cxcywh(self, xyxy, relative=True):
    if xyxy.ndim == 1: xyxy = xyxy.reshape(1, 4)
    bx, by = np.round(xyxy[:,[0,2]].sum(-1)/2), np.round(xyxy[:,[1,3]].sum(-1)/2)
    bw, bh = np.abs(xyxy[:,2]-xyxy[:,0]), np.abs(xyxy[:,3]-xyxy[:,1])
    w, h = self.background_size
    bw = np.minimum(bw, np.minimum(bx, w - bx) * 2)
    bh = np.minimum(bh, np.minimum(by, h - by) * 2)
    if relative:
      bx, bw = bx / w, bw / w
      by, bh = by / h, bh / h
    return np.stack([bx, by, bw, bh], -1)
  
  def build(self, save_path="", verbose=False, show_box=False):
    img = self.background
    cls = set()
    self.unit_list = sorted(self.unit_list, key=lambda x: (x.level, x.xy_cell[1]))
    box, mask, unit_avail = [], np.zeros(img.shape[:2]), []  # return box, union of images, available units
    for u in self.unit_list[::-1]:  # reverse order for NMS
      ratio = self._intersect_ratio_with_mask(u, mask)
      if u.cls_name in tower_unit_list:
        if ratio > self.tower_intersect_ratio_thre: continue
      elif ratio > self.intersect_ratio_thre: continue
      u.draw_mask(mask)
      cls.add(u.cls)
      unit_avail.append(u)
      if u.cls != -1:
        box.append((*u.xyxy, *u.states, u.cls))
    for u in unit_avail[::-1]:  # increase order for drawing
      u.draw(img)
    box = np.array(box, np.float32)
    box[:,:4] = self.xyxy2cxcywh(box[:,:4])
    x = img
    img = Image.fromarray(img)
    if show_box:
      cls2color = build_label2colors(list(cls))
      for u in unit_avail:
        img = u.show_box(img, cls2color)
    if len(save_path): img.save(save_path)
    if verbose: img.show()
    return x, box
  
  def join(self, unit: Unit):
    assert isinstance(unit, Unit), "The join element must be the instance of Unit"
    self.unit_list.append(unit)
  
  def _build_unit_from_path(
      self, path: str | Path,
      xy_bottom_center: tuple,
      level: int,
      max_width: float | tuple = None,  # relative to cell
      join: bool = True,
    ):
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
    unit = Unit(img=img, xy_bottom_center=xy_bottom_center, level=level, background_size=self.background_size, name=name, augment=self.augment)
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
      king0 = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='king-tower', regex='king-tower_0'))
      unit = self._build_unit_from_path(king0, self.bc_pos['king0'], 1)
      self._add_component(unit)
      king1 = self._sample_elem(self.path_manager.search(subset='images', part='segment', name='king-tower', regex='king-tower_1'))
      unit = self._build_unit_from_path(king1, self.bc_pos['king1'], 1)
      self._add_component(unit)
    if queen:
      for i in range(2):  # is enermy?
        for j in range(2):  # left or right
          queen = self._sample_elem(
            self.path_manager.search(subset='images', part='segment', name='queen-tower', regex=f'queen-tower_{i}') +
            self.path_manager.search(subset='images', part='segment', name='cannoneer-tower', regex=f'cannoneer-tower_{i}')
          )
          unit = self._build_unit_from_path(queen, self.bc_pos[f'queen{i}_{j}'], 1)
          unit = self._add_component(unit)
          
  @staticmethod
  def _update_map(
    a: np.ndarray,
    xy: np.ndarray | Sequence,
    size: int = 3, replace: bool = True
  ) -> np.ndarray:
    """
    Update the distribution map after use the `xy` position, modify policy is:
    confirm the center point `xy`, find the `(size, size)` area around the center,
    let `a[*xy] <- a[*xy] / 2`, around values add `a[*xy] / 2 / num_round`,
    for example (left matrix is `a`, let `xy=(1,2), size=3`):
    ```
    [[1. 1. 0.]     [[1.    1.125 0.   ]
      [0. 1. 1.]  ->  [0.    1.125 0.5  ]
      [0. 1. 1.]]     [0.    1.125 1.125]]
    ```

    Args:
      a (np.ndarray): The distribution map (not necessary sum to 1).
      xy (np.ndarray | Sequence): The point to place the unit.
      size (int): The size of the squre width.
      replace (bool): Replace the array `a` directly.
    Return:
      a (np.ndarray): The distribution map after updating.
    """
    if not replace: a = a.copy()
    if not isinstance(xy, np.ndarray):
      xy = np.array(xy)
    assert a[xy[0],xy[1]] != 0 and xy.size == 2

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
    self._update_map(map, xy, size=self.map_cfg['update_size'])
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
    components = [key for key, val in component2unit.items() if unit.cls_name in val]
    if len(components) == 0: return
    cs = []
    for c, prob in important_components:
      if c in components and c not in cs and random.random() < prob:
        components.remove(c)
        cs.append(c)
    for ns, prob in component_prob.items():
      if unit.cls_name in ns: break
    if random.random() < prob:
      k = random.randint(1, len(components))
      cs += self._sample_elem(components, k=k, get_elem=False)
    if not len(cs): return  # low prob and no important components
    for c in cs:
      if isinstance(c, tuple):
        if random.random() < 0.8: c = c[0]  # 0.8 prob for 'bar'
        else: c = c[1]  # 0.2 prob for 'bar-level'
      if c in component_cfg: cfg = component_cfg[c]
      else: cfg = component_cfg[c+str(unit.states[0])]
      center, dx_range, dy_range, max_width = cfg
      if center == 'bottom_center': center = unit.xy_cell
      elif center == 'top_center': center = (unit.xy_cell[0], (unit.xyxy[1]-xyxy_grids[1])/cell_size[1])
      # elif center == 'top_center': center = (unit.xy_cell[0], unit.xyxy[1]/cell_size[1])
      xy = self._sample_from_center(center, dx_range, dy_range)
      path = self.path_manager.path / "images/segment" / c
      if 'bar' in c:  # determine the side 0/1
        paths = list(path.glob(f"{c}_{unit.states[0]}*"))
      else:
        paths = list(path.glob('*'))
      if len(paths) == 0: return
      level = unit2level[c]
      self._build_unit_from_path(self._sample_elem(paths), xy, level, max_width)
  
  def _add_item(self):
    """
    Add background items in `background_item_list`, `big-text`, `emote`.
    Look at `item_cfg.keys()`.
    """
    # (prob, [center, dx_range, dy_range, width_range, max_num]*n)
    for name, (prob, cfgs) in item_cfg.items():
      if random.random() > prob: continue
      for cfg in cfgs:
        if name not in background_item_list:
          paths = list((self.path_segment / name).glob('*'))
        else:
          paths = list((self.path_segment / 'background-items').glob(name+'*'))
        level = unit2level[name]  # [0: background_items, 3: big-text, emote]
        center, dx_range, dy_range, w_range, maxn = cfg
        if maxn == 1:
          n = self._sample_elem(range(2))
        else: n = self._sample_elem(range(maxn)) + 1
        for _ in range(n):
          xy = self._sample_from_center(center, dx_range, dy_range)
          path = self._sample_elem(paths)
          self._build_unit_from_path(path, xy, level, w_range)
  
  def add_unit(self, n=1):
    """
    Add unit in [ground, flying, others] randomly.
    Unit list looks at `katacr/constants/label_list.py`
    """
    self._add_item()
    paths = []
    for p in (self.path_segment).glob('*'):
      if p.name in [
        'backgrounds', 'king-tower', 'queen-tower', 'cannoneer-tower',
      ] + drop_units:
        continue
      paths.append(p)
    for _ in range(n):
      p: Path = self._sample_elem(paths)
      level = unit2level[p.name]  # [1, 2, 3]
      xy = self._sample_from_map(level)
      unit = self._build_unit_from_path(self._sample_elem(list(p.glob('*'))), xy, level)
      self._add_component(unit)
  
  def reset(self):
    self.build_background()
    self.unit_list = []

if __name__ == '__main__':
  generator = Generator(seed=42, intersect_ratio_thre=0.5, augment=True)
  path_generation = path_logs / "generation"
  path_generation.mkdir(exist_ok=True)
  for i in range(5):
    # generator = Generator(background_index=None, seed=42+i, intersect_ratio_thre=0.9)
    generator.add_tower()
    generator.add_unit(n=30)
    x, box = generator.build(verbose=False, show_box=True, save_path=str(path_generation / f"test{10+2*i}.jpg"))
    generator.build(verbose=False, show_box=False, save_path=str(path_generation / f"test{10+2*i+1}.jpg"))
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
