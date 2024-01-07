from PIL import Image
from pathlib import Path
from katacr.utils.detection import plot_cells_PIL, plot_box_PIL, build_label2colors
from typing import Tuple, List
import numpy as np
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.build_dataset.constant import xyxy_grids, path_logs, bottom_center_grid_position
from katacr.constants.label_list import unit2idx, idx2unit
from katacr.constants.state_list import state2idx, idx2state
from katacr.build_dataset.generation_config import map_fly, map_ground, level2units, unit2level, grid_size
import random

path_save = path_logs / "background_cells"
path_save.mkdir(exist_ok=True)
cell_size = np.array([(xyxy_grids[3] - xyxy_grids[1]) / grid_size[1], (xyxy_grids[2] - xyxy_grids[0]) / grid_size[0]])[::-1]  # cell pixel: (w, h)

def cell2pixel(xy: tuple):
  if type(xy) != np.ndarray: xy = np.array(xy)
  return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

def show_point(img: Image, xy_cell: tuple):
  xy = cell2pixel(xy_cell)
  img = plot_box_PIL(img, (xy[0],xy[1],5,5), draw_center_point=True)
  return img

class Unit:
  def __init__(
      self, img: np.ndarray,
      xy_bottom_center: tuple | np.ndarray,
      level: int,
      background_size: Tuple[int],
      name: str | None = None,
      cls: str | int | None = None,
      states: str | np.ndarray = None,
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
      states (str | np.ndarray): The states of the unit.
      background (bool): The image is the background.
    """
    self.xy_cell = xy_bottom_center
    h, w = img.shape[:2]
    xy = cell2pixel(self.xy_cell)
    self.xyxy = np.array((
      max(xy[0] - w // 2, 0),
      max(xy[1] - h, 0),
      min(xy[0] + (w + 1) // 2, background_size[0]),
      min(xy[1], background_size[1])
    ), np.int32)  # xyxy relative to background
    size = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
    if size / (h * w) < 0.2:
      self.xyxy = np.zeros(4)

    # clip the image
    xyxy_relative = (self.xyxy - np.array([xy[0] - w // 2, xy[1] - h] * 2)).astype(np.int32)
    self.xyxy_relative = xyxy_relative
    img = img[xyxy_relative[1]:xyxy_relative[3], xyxy_relative[0]:xyxy_relative[2], :]
    if img.shape[-1] == 4:
      self.mask = img[...,3] > 0
      self.img = img[...,:3]
    else:
      self.mask = img.sum(-1) > 0
      self.img = img

    if name is not None or (cls is not None and states is not None):
      if name is not None:
        cls, *states = name.split('_')
      self.cls = unit2idx[cls] if isinstance(cls, str) else cls
      if isinstance(states, np.ndarray):
        self.states = states
      else:
        self.states = np.zeros(7, np.int32)
        for s in states:
          c, i = state2idx[s]
          self.states[c] = i
    else:
      raise "Error: You must give the label of the unit (when not background)."
    self.level = level

  def get_name(self):
    name = idx2unit[self.cls]
    for i, s in enumerate(self.states):
      if i == 0:
        name += idx2state[s]
      elif s != 0:
        name += '_' + idx2state[i*10+s]
    return name
  
  def draw(self, img: np.ndarray, inplace=True):
    if not inplace: img = img.copy()
    try:
      img[self.xyxy[1]:self.xyxy[3],self.xyxy[0]:self.xyxy[2],:][self.mask] = self.img[self.mask]
    except:
      print(self.xyxy, self.mask.shape, self.img.shape, self.xy_cell)
      raise "GG"
    return img

  def show_box(self, img: Image, cls2color: dict | None = None):
    color = cls2color[self.cls] if cls2color is not None else 'red'
    return plot_box_PIL(img, self.xyxy, text=self.get_name(), format='voc', box_color=color)

class Generator:
  bc_pos: dict = bottom_center_grid_position  # with keys: ['king0', 'king1', 'queen0_0', 'queen0_1', 'queen1_0', 'queen1_1']

  def __init__(
      self, background_index: int | None = None,
      unit_list: Tuple[Unit,...] = None,
      fliplr: float = 0.5,
      seed: int | None = None,
      intersect_ratio_thre: float = 0.8
    ):
    self.path_manager = PathManager()
    self.background_index = background_index
    self.build_background()
    self.unit_list = [] if unit_list is None else unit_list
    self.fliplr = fliplr  # The probability of fliping the image left and right
    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)
    self.intersect_ratio_thre = intersect_ratio_thre
  
  def build_background(self):
    background_index = self.background_index
    if background_index is None:
      n = len(self.path_manager.search(subset='images', part='segment', name='backgrounds', regex="background\d+.jpg"))
      background_index = random.randint(1, n)
    self.background = np.array(Image.open(self.path_manager.path / f"images/segment/backgrounds/background{background_index:02}.jpg"))
    self.background_size = self.background.shape[:2][::-1]
    assert self.background.shape[0] == 896 and self.background.shape[1] == 568
  
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
    box, unit_avail = [], []
    box_xyxy = np.array([u.xyxy for u in self.unit_list])
    for u in self.unit_list:
      box_residue = box_xyxy[~(box_xyxy == u.xyxy.reshape(1,4)).all(-1)]
      if (
        (len(box_residue) and self._max_intersect_ratio(u.xyxy, box_residue) > self.intersect_ratio_thre) or
        u.xyxy[2] - u.xyxy[0] < 6 or
        u.xyxy[3] - u.xyxy[1] < 6
      ):
        continue
      u.draw(img)
      cls.add(u.cls)
      unit_avail.append(u)
      box.append((*u.xyxy, *u.states, u.cls))
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
  
  def build_unit_from_path(self, path: str | Path, xy_bottom_center: tuple, level: int, join: bool = True):
    img = np.array(Image.open(path))
    if random.uniform(0, 1) < self.fliplr:
      img = np.fliplr(img)
    name = path.name.rsplit('.',1)[0]
    name = name.rsplit('_',1)[0]
    unit = Unit(img=img, xy_bottom_center=xy_bottom_center, level=level, background_size=self.background_size, name=name)
    if join: self.unit_list.append(unit)
    return unit
  
  @staticmethod
  def _sample_one(x):
    return random.sample(x, k=1)[0]
  
  def add_tower(self, king=True, queen=True):
    if king:
      king0 = self._sample_one(self.path_manager.search(subset='images', part='segment', name='king-tower', regex='king-tower_0'))
      self.build_unit_from_path(king0, self.bc_pos['king0'], 1)
      king1 = self._sample_one(self.path_manager.search(subset='images', part='segment', name='king-tower', regex='king-tower_1'))
      self.build_unit_from_path(king1, self.bc_pos['king1'], 1)
    if queen:
      for i in range(2):  # is enermy?
        for j in range(2):  # left or right
          queen = self._sample_one(
            self.path_manager.search(subset='images', part='segment', name='queen-tower', regex=f'queen-tower_{i}') +
            self.path_manager.search(subset='images', part='segment', name='cannoneer-tower', regex=f'cannoneer-tower_{i}')
          )
          self.build_unit_from_path(queen, self.bc_pos[f'queen{i}_{j}'], 1)
  
  @staticmethod
  def _sample_from_map(map: List[List[int]] | np.ndarray, noisy: bool = True):
    """
    Sample one position with non-zero element from the 2d boolean map array.
    
    Args:
      map (List[List[int]] | np.ndarray): The 0/1 map in `katacr/build_dataset/generation_config.py`.
      noisy (bool): Whether add Gaussian noisy (mu=0, sigma=0.4) to the position.
    """
    if not isinstance(map, np.ndarray): map = np.array(map)
    map = map.astype(np.bool_)
    nonzeros = np.argwhere(map)
    xy = nonzeros[np.random.choice(nonzeros.shape[0], 1)[0]][::-1].astype(np.float32)
    xy += 0.5  # move to center of the cell
    if noisy:
      xy += np.random.randn(2) * 0.4
      xy = np.array([np.clip(xy[0], 0, grid_size[0]), np.clip(xy[1], 0, grid_size[1])])
    return xy
  
  def add_unit(self, n=1):
    """
    Add unit in [ground, flying, others] randomly.
    Unit list looks at `katacr/constants/label_list.py`
    """
    paths = []
    for p in (self.path_manager.path / "images/segment").glob('*'):
      if p.name in ['backgrounds', 'king-tower', 'queen-tower', 'cannoneer-tower']: continue
      paths.append(p)
    for _ in range(n):
      p: Path = self._sample_one(paths)
      level = unit2level[p.name]  # [1, 2, 3]
      map = map_ground if level == 0 else map_fly
      xy = self._sample_from_map(map)
      self.build_unit_from_path(self._sample_one(list(p.glob('*'))), xy, level)
  
  def reset(self):
    self.build_background()
    self.unit_list = []

if __name__ == '__main__':
  generator = Generator(seed=42, intersect_ratio_thre=0.8)
  for i in range(10):
    # generator = Generator(background_index=None, seed=42+i, intersect_ratio_thre=0.9)
    generator.add_tower()
    generator.add_unit(n=100)
    generator.build(verbose=False, show_box=True, save_path=f"/home/yy/Pictures/test{2*i}.jpg")
    generator.build(verbose=False, show_box=False, save_path=f"/home/yy/Pictures/test{2*i+1}.jpg")
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
