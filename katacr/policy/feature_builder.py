"""
Build state, reward, action from VisualFusion result.
Main functions:
1. Build offline RL dataset based on 5fps, 0.2it/s, interval=2 (VisualFusion 10fps, 0.1it/s)
2. Build state, reward for validating.
"""

import scipy.sparse
import scipy.spatial
from katacr.yolov8.custom_result import CRResults
from queue import Queue
import numpy as np
import warnings
from katacr.constants.label_list import idx2unit, unit2idx, tower_bar2_unit_list, moveable_bar2_unit_list
from katacr.policy.utils import extract_img, pixel2cell, cell2pixel, xyxy2center, background_size
from katacr.build_dataset.generation_config import except_king_tower_unit_list, except_spell_and_object_unit_list, spell_unit_list
from katacr.policy.visual_fusion import VisualFusion
from typing import List
from pathlib import Path
import itertools
import cv2
import scipy

"""
Statet data UnitInfo:
xy (ndarray): center of body image, relative to cell
cls (int): class of the unit
bel (int): belong of the unit
body (ndarray): unit body image
bar1 (ndarray): unit first bar
bar2 (ndarray): unit second bar
"""
BASE_UNIT_INFO = dict(xy=None, cls=None, bel=None, body=None, bar1=None, bar2=None)
BAR_CENTER2BODY_DELTA_Y = 40
DIS_BAR_AND_BAR_LEVEL_THRE = 5  # the threshold of distance between bar and bar-level
DIS_BAR_AND_BODY_THRE = 20  # the threshold of distance between bar and body

class StateBuilder:
  bar1_units = ['bar', 'tower-bar', 'king-tower-bar']
  bar2_units = [ 'dagger-duchess-tower-bar', 'skeleton-king-bar']

  def __init__(self, persist: int=3):
    """
    Args:
      persist (int): The maximum time to memory in bar_history.
    Variables:
      bar2xywhc (dict): key (int)=bar track_id,
        val (ndarray)=(relative xy to body image left top, wh of last body image,
        class of last body image)
      bar_history (queue): val=(bar track_id, time)
    """
    self.persist = persist
    self.bar2xywhct = {}
    self.bar_history = Queue()
    self.bar_items: List[BarItem] = []
  
  def _add_bar_item(self, bar_level=None, bar1=None, bar2=None, body=None):
    self.bar_items.append(BarItem(self, bar_level=bar_level, bar1=bar1, bar2=bar2, body=body))
  
  def debug(self, verbose=False):
    from PIL import ImageDraw, Image
    rimg = Image.fromarray(self.arena.show_box()[...,::-1])
    for bar_item in self.bar_items:
      info = bar_item.debug()
      x, y = cell2pixel(info['xy'])
      print("Draw", x, y)
      draw = ImageDraw.Draw(rimg)
      draw.rounded_rectangle([x-2,y-2,x+2,y+2], radius=3, fill=(255,0,0))
    if verbose:
      rimg.show()
    return np.array(rimg)[...,::-1]
  
  def _build_bar_items(self):
    """
    Build BarItem (consider parts: bar-level, bar1, body):
    1. King Tower: king-tower-bar with king-tower
    2. Defense Tower: tower-bar and tower-bar2 with defense-tower (queen_tower, cannoneer_tower, dagger-duchess-tower)
    3. bar-level with bar
    4. bar and bar-level singly (unmatched)
    """
    self.bar_items = []
    get_unit_box = lambda unit: self.box[self.box[:,-2] == unit2idx[unit]]
    ### King Tower ###
    king_tower_bar_box = get_unit_box('king-tower-bar')
    if len(king_tower_bar_box):
      for b in king_tower_bar_box:
        xy = xyxy2center(b[:4])
        assert (b[-1] == 0) ^ (xy[1] < background_size[1] / 2), f"The king_tower_bar_box has bel={b[-1]}, but the center of box={xy} is not correct side"
        king_tower_box = self.box[(self.box[:,-2] == unit2idx['king-tower']) & (self.box[:,-1] == b[-1])]
        assert len(king_tower_box) == 1, f"King tower number must have one, when king_tower_bar is existed"
        if len(king_tower_box):
          self._add_bar_item(bar1=b, body=king_tower_box[0])
    ### Defense Tower ###
    tower_bar_box = get_unit_box('tower-bar')
    tower_bar2_box = get_unit_box('dagger-duchess-tower-bar')
    tower_box = np.concatenate([get_unit_box(name) for name in except_king_tower_unit_list], 0)
    # left top: 0, right top: 1, left bottom: 2, right bottom: 3
    patches = [[None, None, None] for _ in range(4)]  # each value: (bar1_box, bar2_box, tower_box)
    for i, boxes in enumerate((tower_bar_box, tower_bar2_box, tower_box)):
      for b in boxes:
        xy = xyxy2center(b[:4])
        idx = 2 * int(xy[1] > background_size[1] / 2) + int(xy[0] > background_size[0] / 2)
        patches[idx][i] = b
    for i, patch in enumerate(patches):
      assert not ((patch[0] is None) ^ (patch[2] is None)), f"The tower_bar and tower misalign at patch={i}"
      if patch[0] is not None:
        self._add_bar_item(bar1=patch[0], bar2=patch[1], body=patch[2])
    ### bar-level and bar1 ###
    bar_level_box = get_unit_box('bar-level')
    bar_level_mask = np.ones((len(bar_level_box),), np.bool_)  # available bar-level indexs
    bar_box = get_unit_box('bar')
    bar_mask = np.ones((len(bar_box),), np.bool_)  # available bar indexs
    xy1 = np.stack([bar_level_box[:,2], np.mean(bar_level_box[:,[1,3]], -1)], -1)  # right center
    xy2 = np.stack([bar_box[:,0], np.mean(bar_box[:,[1,3]], -1)], -1)  # left center
    dis = scipy.spatial.distance.cdist(xy2, xy1)
    # bar find bar-level
    for i, bar in enumerate(bar_box):
      idx = np.argmin(dis[i])
      if dis[i,idx] < DIS_BAR_AND_BAR_LEVEL_THRE:
        assert bar_level_mask[idx], f"The bar(id={bar[-4]})'s nearest bar-level(id={bar_level_box[idx,-4]}) has been used"
        bar_level_mask[idx] = bar_mask[i] = False
        self._add_bar_item(bar_level=bar_level_box[idx], bar1=bar)
    ### Add remaining bar and bar-level singly ###
    for name, box, mask in (('bar_level', bar_level_box, bar_level_mask), ('bar1', bar_box, bar_mask)):
      for i in np.argwhere(mask).reshape(-1):
        self._add_bar_item(**{name: box[i]})
  
  def _combine_bar_items(self):
    """
    Combine body image and bar2 (except tower-bar2) with BarItem.
    """
    xy_bars = np.array([i.center for i in self.bar_items], np.float32)
    mask_bars = np.ones((len(xy_bars),), np.bool_)
    no_bar_box = []
    moveable_box = []
    for box in self.box:
      cls = box[-2]
      cls_name = idx2unit[int(cls)]
      if cls_name in spell_unit_list:
        no_bar_box.append(box)
      elif cls_name in (except_spell_and_object_unit_list + moveable_bar2_unit_list):
        moveable_box.append(box)
    moveable_box = np.array(moveable_box, np.float32)
    moveable_box = moveable_box[np.argsort(moveable_box[:,1])[::-1]]  # decrease box by top-y
    for box in moveable_box:
      xyxy = box[:4]
      datum = np.array([(xyxy[0]+xyxy[2])/2, xyxy[1]], np.float32).reshape(1, 2)  # top center
      xy_bars_avail = xy_bars[mask_bars]  # body image consider unused bar_item
      idx_map = np.argwhere(mask_bars).reshape(-1)
      cls_name = idx2unit[int(box[-2])]
      if cls_name == 'skeleton-king-bar':  # bottom center
        datum[0,1] = xyxy[3]
        xy_bars_avail = xy_bars  # bar2 consider all bar_item
        idx_map = np.arange(len(mask_bars))
      if len(xy_bars_avail) == 0: continue
      dis = scipy.spatial.distance.cdist(datum, xy_bars_avail)[0]
      print(f"cls={cls_name},id={box[-4]},box dis=", dis)
      idx = np.argmin(dis)
      if dis[idx] < DIS_BAR_AND_BODY_THRE:
        bar_item = self.bar_items[idx_map[idx]]
        if cls_name in moveable_bar2_unit_list:  # bar2
          assert bar_item.bar2 is None, f"The bar1(id={bar_item.bar1[-4]}) has bar2(id={bar_item.bar2[-4]}), but bar2(id={box[-4]}) finds it again."
          bar_item.bar2 = box
        else:  # body
          assert bar_item.body is None
          bar_item.body = box
          mask_bars[idx] = False
      else:
        no_bar_box.append(box)
    for box in no_bar_box:
      self._add_bar_item(body=box)
  
  def _update_bar_items_history(self):
    while not self.bar_history.empty() and self.bar_history.queue[0][1] - self.time > self.persist:
      idx, _ = self.bar_history.get()
      if self.bar2xywhct[idx][-1] - self.time > self.persist:  # check latest time
        self.bar2xywhct.pop(idx)
    for bar in self.bar_items:
      bar.update_state_bar_info(self.time)

  def update(self, info: dict):
    """
    Args:
      info (dict): The return in `VisualFusion.process()`,
        which has keys=[time, arena, cards, elixir]
    """
    self.time: int = info['time']
    self.arena: CRResults = info['arena']
    self.cards: dict = info['cards']
    self.elixir: int = info['elixir']
    self.box = box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    assert box.shape[-1] == 8, f"The last dim should be 8, but get {box.shape[-1]}"
    ### Step 1: Build bar items ###
    self._build_bar_items()
    ### Step 2: Combine units and bar2 with their BarItem ###
    self._combine_bar_items()
    ### Step 3: Update bar history ###
    self._update_bar_items_history()

class BarItem:

  def __init__(
      self, state: StateBuilder, bar_level: np.ndarray = None,
      bar1: np.ndarray = None, bar2: np.ndarray = None, body: np.ndarray = None,
      ):
    """
    The bar item which combine relative bar-level, bar1 and bar2,
    the relative bars such as `bar0/1`, `dagger-duchess-tower-bar`,
    `skeleton-king-bar`, etc.

    The box's shape=(8,), each value means (xyxy, track_id, conf, cls, bel)

    Args:
      state (StateBuilder): The StateBuilder which is running.
      bar_level (ndarray): The bar-level box on the left of bar, if exist.
      bar1 (ndarray): The main-bar box of the unit, unit_name=[bar, tower-bar, king-tower-bar].
      bar2 (ndarray): The sub-bar box of the unit, unit_name=[dagger-duchess-tower-bar, skeleton-king-bar].
      body (ndarray): The body box of the unit, unit_name=tower_unit_list+ground_unit_list+fly_unit_list
    Functions:
      update_bar2xywhct: Update `bar2xywhct` in StateBuilder for every bars in BarItem.
    """
    for i in [bar_level, bar1, bar2, body]:
      if i is not None: assert i.ndim == 1 and i.shape[0] == 8, "The box dim should be 1, shape=(8,)"
    self.bar_level, self.bar1, self.bar2 = bar_level, bar1, bar2
    self.body, self.bar2xywhct, self.bar_history = body, state.bar2xywhct, state.bar_history
    self.bars = [self.bar_level, self.bar1, self.bar2]
    self.bel = None
    for i, bar in enumerate(self.bars + [body]):
      if bar is not None and i != 2:  # Don't consider bar2 belong
        bel = bar[-1]
        if self.bel is not None:
          assert self.bel == bel, "all the bars must have same belong"
        self.bel = bel
    assert self.bel is not None, "The belong of BarItem is None"
  
  def debug(self):
    info = self.get_unit_info(np.empty(background_size[::-1], np.int32))
    cls_name = idx2unit[info['cls']] if info['cls'] is not None else None
    print(f"Bar Items (cls={cls_name}) (id): ", end='')
    for name in ('bar_level', 'bar1', 'bar2', 'body'):
      box = getattr(self, name)
      if box is not None:
        print(f"{name}={box[-4]}", end=' ')
    print()
    return info
  
  def update_state_bar_info(self, time):
    if self.body is None: return
    xyxy = self.body[:4]
    cls = self.body[-2]
    xy_body, wh_body = xyxy[:2], xyxy[2:] - xyxy[:2]
    for bar in self.bars:
      if bar is None: continue
      xy_bar = bar[:2]
      id = int(bar[-4])
      if id in self.bar2xywhct:
        last_cls = self.bar2xywhct[id][-2]
        if last_cls != cls:
          warnings.warn(f"Last class is {idx2unit[int(last_cls)]} but new class is {idx2unit[int(cls)]} at time {time}")
      self.bar2xywhct[id] = np.concatenate([xy_body - xy_bar, wh_body, [cls, time]]).astype(np.int32)
      self.bar_history.put((id, time))
  
  def get_unit_info(self, img):
    """
    Parse unit_info from bars and body image.

    Args:
      img: The arena image (part2).
    Returns:
      info: The unit info same as `BASE_UNIT_INFO` struct.
    """
    last_time = -1
    info = BASE_UNIT_INFO.copy()
    if self.body is None:
      for bar in self.bars:
        if bar is None: continue
        id = int(bar[-4]) 
        if id in self.bar2xywhct:
          xywhct = self.bar2xywhct[id]
          if xywhct[-1] <= last_time: continue
          last_time = xywhct[-1]
          xyxy_box = bar[:4]
          xy = xywhct[:2] + xyxy_box[:2]
          info['xy'] = pixel2cell([xy[0]+xywhct[2]/2, xy[1]+xywhct[3]/2])
          info['cls'] = int(xywhct[-2])
        else:
          xy = self.center
          xy[1] += BAR_CENTER2BODY_DELTA_Y
          info['xy'] = pixel2cell(xy)
    else:
      xyxy = self.body[:4]
      info['xy'] = pixel2cell(xyxy.reshape(2,2).mean(0))
      info['cls'] = int(self.body[-2])
      info['body'] = extract_img(img, xyxy)
    for name in ['bar1', 'bar2']:
      bar = getattr(self, name)
      info[name] = extract_img(img, bar[:4]) if bar is not None else None
    info['bel'] = self.bel
    return info
  
  @property
  def center(self):
    """
    center of bar-level and bar1, if they exist.
    """
    xyxy = np.array([np.inf, np.inf, 0, 0], np.float32)
    for bar in [self.bar_level, self.bar1]:
      if bar is None: continue
      xyxy[:2] = np.minimum(xyxy[:2], bar[:2])
      xyxy[2:] = np.maximum(xyxy[2:], bar[2:4])
    return xyxy.reshape(2,2).mean(0)
