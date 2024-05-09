"""
Build state from VisualFusion result.
Main functions:
1. Build offline RL dataset based on 5fps, 0.2it/s, interval=2 (VisualFusion 10fps, 0.1it/s)
2. Build state, reward for validating.
State data info (Dict):
- 'time' (int): Passed time (second)
- 'unit_infos' (List): [UnitInfo]
  UnitInfo:
    - xy (ndarray): center of body image, relative to cell
    - cls (int): class of the unit
    - bel (int): belong of the unit
    - body (ndarray): unit body image (Don't use now)
    - bar1 (ndarray): unit first bar
    - bar2 (ndarray): unit second bar
- 'cards' (List): Card index based on classifier.card2idx (length 5: next card, card1~4)
- 'elixir' (int): Current elixir
"""

import scipy.sparse
import scipy.spatial
from katacr.yolov8.custom_result import CRResults
from queue import Queue
import numpy as np
from katacr.constants.label_list import idx2unit, unit2idx, bar2_unit_list
from katacr.policy.perceptron.utils import extract_img, pixel2cell, cell2pixel, xyxy2center, background_size, xyxy2sub, pil_draw_text, LOW_ALPHA, xyxy2topcenter, edit_distance
from katacr.build_dataset.generation_config import except_king_tower_unit_list, except_spell_and_object_unit_list, spell_unit_list, object_unit_list
from typing import List, Dict, Tuple
import scipy
from collections import Counter, defaultdict
from katacr.ocr_text.paddle_ocr import OCR
from katacr.constants.card_list import card2idx, unit2cards

BASE_UNIT_INFO = dict(xy=None, cls=None, bel=None, body=None, bar1=None, bar2=None)
BAR_CENTER2BODY_DELTA_Y = 40
DIS_BAR_AND_BAR_LEVEL_THRE = 15  # the threshold of distance between bar and bar-level
DIS_BAR_AND_BODY_THRE = 35  # the threshold of distance between bar and body
SUB_XYXY_KING_TOWER_BAR = (0.22, 0.23, 1.0, 0.8)
SUB_XYXY_TOWER_BAR = [(0.26, 0.2, 1.0, 0.8), (0.26, 0.35, 1.0, 0.9)]
REMOVE_DEPLOY_DIS_THRE = 50  # dis between the bottom center of text and top center of unit
DEPLOY_HISTORY_PERSIST_FRAME = 15  # 15 * 0.1 = 1.5 sec
EDIT_DISTANCE_THRE = 2  # Levenshtein distance between ocr text **in** target text

class StateBuilder:
  bar1_units = ['bar', 'tower-bar', 'king-tower-bar']
  bar2_units = [ 'dagger-duchess-tower-bar', 'skeleton-king-bar']

  def __init__(self, persist: int = 3, ocr: OCR = None):
    """
    Args:
      persist (int): The maximum time to memory in bar_history (second).
    Variables:
      bar2xywhct (dict): key (int)=bar track_id,
        val (ndarray)=(relative xy to body image left top, wh of last body image,
        class of last body image, time)
      bar_history (queue): val=(bar track_id, time)
      bar_items (BarItems): Memory all unit items (bar-level, bar1, bar2, body).
      bel_memory (dict): key (int)=bar track_id,
        val (tuple)=(count of bel=0, count of bel=1)
      cls_memory (dict): key (int)=bar track_id,
        val (Counter)=count of each class
      text_info (list): val=(text_class_name, x, y),
        text_class_name is the name of text, xy is bottom center of text 
      deploy_history (queue): The history of deploy cards.
    """
    self.persist = persist
    self.ocr = OCR(lang='en') if ocr is None else ocr
    self.reset()

  def reset(self):
    self.time = 0
    self.bar2xywht = dict()
    self.frame_count = 0
    self.bar_history = Queue()
    self.bar_items: List[BarItem] = []
    self.bel_memory: Dict[int,Counter] = defaultdict(Counter)
    self.cls_memory: Dict[int,Counter] = defaultdict(Counter)
    self.deploy_history = dict()
  
  def _add_bar_item(self, bar_level=None, bar1=None, bar2=None, body=None):
    self.bar_items.append(BarItem(self, bar_level=bar_level, bar1=bar1, bar2=bar2, body=body))
  
  def render(self, action=None):
    from PIL import ImageDraw, Image
    from katacr.policy.replay_data.data_display import GridDrawer, DISPLAY_SCALE, build_label2colors
    rimg = Image.fromarray(self.arena.show_box()[...,::-1])
    state = self.get_state()
    arena = GridDrawer()
    draw = ImageDraw.Draw(rimg)
    label2color = build_label2colors([bar['cls'] for bar in state['unit_infos'] if bar['cls'] is not None])
    label2color[None] = (255, 255, 255)
    for info in state['unit_infos']:
      arena.paint(arena.find_near_pos(info['xy']*DISPLAY_SCALE), label2color[info['cls']], info['bel'])
      x, y = cell2pixel(info['xy'])
      draw.rounded_rectangle([x-2,y-2,x+2,y+2], radius=3, fill=(255,0,0))
    if action is not None and action['xy'] is not None:
      text = f"{action['card_id']}"
      if 'offset' in action:
        text += f"\n-{action['offset']}"
      arena.paint(action['xy'].astype(np.int32), (255,236,158), text, rect=False, circle=True, text_color=(0,0,0))
    elixir_and_cards = f"elixir: {self.elixir}\n{self.cards}"
    rimg = pil_draw_text(rimg, (0, rimg.size[1]), elixir_and_cards, font_size=14, text_pos='left down')
    # rimg = pil_draw_text(rimg, (self.parts_pos[0,0]-self.parts_pos[1,0], 0), f"Time: {self.time}", font_size=20, text_pos='right top')
    rimg = pil_draw_text(rimg, (500, 0), f"Time: {self.time}", font_size=20, text_pos='right top')
    ret = np.concatenate([np.array(rimg), arena.image], 1)
    return ret[...,::-1]
  
  def _update_bel_memory(self):
    for b in self.box:
      id = int(b[-4])
      counter = self.bel_memory[id]
      counter.update({int(b[-1]): 1})
      mxid = counter.most_common(1)[0][0]
      if mxid != int(b[-1]):
        print(f"Warning(state): (time={self.time}) Find id={id} has wrong bel={int(b[-1])}, change to {mxid}")
        b[-1] = mxid
  
  def _build_bar_items(self):
    """
    Build BarItem (consider parts: bar-level, bar1, body):
    1. King Tower: king-tower-bar with king-tower
    2. Defense Tower: tower-bar and tower-bar2 with defense-tower (queen_tower, cannoneer_tower, dagger-duchess-tower)
    3. bar-level with bar
    4. bar and bar-level singly (unmatched)
    5. add part2 (expect tower-bar2)
    """
    self.bar_items = []
    get_unit_box = lambda unit: self.box[self.box[:,-2] == unit2idx[unit]]
    ### King Tower ###
    king_tower_box = get_unit_box('king-tower')
    king_tower_bar_box = get_unit_box('king-tower-bar')
    patches = [[None, None] for _ in range(2)]
    for i, boxes in enumerate((king_tower_bar_box, king_tower_box)):
      for b in boxes:
        xy = xyxy2center(b[:4])
        bel = int(b[-1])
        if (bel == 0 and xy[1] > 720) or (bel == 1 and xy[1] < 210):
          patches[bel][i] = b
    for patch in patches:
      if patch[0] is not None or patch[1] is not None:
        self._add_bar_item(bar1=patch[0], body=patch[1])
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
      if (patch[0] is None) ^ (patch[2] is None):
        print(f"Warning: (time={self.time}) The tower_bar and tower misalign at patch={i}")
        continue
      if patch[0] is not None:
        self._add_bar_item(bar1=patch[0], bar2=patch[1], body=patch[2])
    ### bar-level and bar1 ###
    bar_level_box = get_unit_box('bar-level')
    bar_level_mask = np.ones((len(bar_level_box),), np.bool_)  # available bar-level indexs
    bar_box = get_unit_box('bar')
    bar_mask = np.ones((len(bar_box),), np.bool_)  # available bar indexs
    xy1 = np.stack([bar_level_box[:,2], np.mean(bar_level_box[:,[1,3]], -1)], -1)  # right center
    xy2 = np.stack([bar_box[:,0], np.mean(bar_box[:,[1,3]], -1)], -1)  # left center
    if len(xy1) and len(xy2):
      # bar find bar-level
      dis = scipy.spatial.distance.cdist(xy2, xy1)
      for i, bar in enumerate(bar_box):
        # print(f"cls={idx2unit[int(bar[-2])]},id={int(bar[-4])},box dis={dis[i]}")  # DEBUG: distance
        idx = np.argmin(dis[i])
        if dis[i,idx] < DIS_BAR_AND_BAR_LEVEL_THRE and dis[i,idx] == np.min(dis[:,idx]):
          assert bar_level_mask[idx], f"The bar(id={bar[-4]})'s nearest bar-level(id={bar_level_box[idx,-4]}) has been used"
          bar_mask[i] = bar_level_mask[idx] = False
          self._add_bar_item(bar_level=bar_level_box[idx], bar1=bar)
    ### Add remaining bar and bar-level singly ###
    for name, box, mask in (('bar_level', bar_level_box, bar_level_mask), ('bar1', bar_box, bar_mask)):
      for i in np.argwhere(mask).reshape(-1):
        self._add_bar_item(**{name: box[i]})
    ### Add bar2 ###
    xy_bars = np.array([i.center for i in self.bar_items], np.float32)
    for box in self.box:
      cls_name = idx2unit[int(box[-2])]
      if cls_name == 'skeleton-king-bar':  # bottom center
        xyxy = box[:4]
        datum = np.array([(xyxy[0]+xyxy[2])/2, xyxy[3]], np.float32).reshape(1, 2)  # bottom center
        dis = scipy.spatial.distance.cdist(datum, xy_bars)[0]
        idx = np.argmin(dis)
        if dis[idx] < DIS_BAR_AND_BODY_THRE:
          bar_item = self.bar_items[idx]
          if bar_item.bar2 is not None:
            print(f"Warning(state): (time={self.time}) The bar1(id={bar_item.bar1[-4]}) has bar2(id={bar_item.bar2[-4]}), but bar2(id={box[-4]}) finds it again.")
          # assert bar_item.bar2 is None, f"The bar1(id={bar_item.bar1[-4]}) has bar2(id={bar_item.bar2[-4]}), but bar2(id={box[-4]}) finds it again."
          bar_item.bar2 = box
        else:
          self._add_bar_item(bar2=box)
  
  def _find_text_info(self, deploy_cards):
    ### Update deploy history first ###
    for card in deploy_cards:
      self.deploy_history[card] = self.frame_count
    for k, v in list(self.deploy_history.items()):
      if self.frame_count - v >= DEPLOY_HISTORY_PERSIST_FRAME:
        self.deploy_history.pop(k)
    text_info = []
    result = self.ocr(self.img)[0]
    # if 28 <= self.frame_count <= 30:
    #   print(f"frame={self.frame_count}, ocr_result:", result)
    #   print(f"deploy_history={self.deploy_history}, {deploy_cards=}")
    if result is not None:
      for info in result:
        det, rec = info
        rec = ''.join([c for c in rec[0].lower() if c in LOW_ALPHA])
        if len(rec) <= 2: continue
        for name in self.deploy_history:
          tname = name.lower().replace('-', '')
          if edit_distance(rec, tname, dis='s1') <= EDIT_DISTANCE_THRE:
            text_info.append((card2idx[name], (det[2][0]+det[3][0])//2, (det[2][1]+det[3][1])//2))
            # print("Find text info:", name, rec)
    self.text_info = np.array(text_info, np.int32).reshape(-1, 3)
  
  def _combine_bar_items(self):
    """ Combine body image with BarItem. """
    no_body_items = [i for i in self.bar_items if i.body is None]
    xy_bars = np.array([i.center for i in no_body_items], np.float32)
    mask_bars = np.ones((len(xy_bars),), np.bool_)
    no_bar_box = []
    moveable_box = []
    for box in self.box:
      cls = box[-2]
      cls_name = idx2unit[int(cls)]
      is_deploy = False
      if cls_name in unit2cards:
        cards = unit2cards[cls_name]
        for c in cards:
          idx = card2idx[c]
          texts = self.text_info[self.text_info[:,0] == idx]
          if len(texts) == 0: continue
          dis = scipy.spatial.distance.cdist(xyxy2topcenter(box[:4])[None,...], texts[:,1:])
          # print(f"Card {c} with unit id={box[-4]} distance:", dis)  # DEBUG
          if dis.min() < REMOVE_DEPLOY_DIS_THRE:
            is_deploy = True
            # print(f"Skip card {c} with id={box[-4]}")
            break
      if is_deploy: continue
      if cls_name in set(spell_unit_list).union(object_unit_list):
        no_bar_box.append(box)
      elif cls_name in set(except_spell_and_object_unit_list) - set(bar2_unit_list):
        moveable_box.append(box)
    moveable_box = np.array(moveable_box, np.float32)
    if len(moveable_box):
      # moveable_box = moveable_box[np.argsort(moveable_box[:,1])[::-1]]  # decrease box by top-y
      # decrease box by (bel, top-y), combo bel first then combo top-y decreasingly
      sorted_idx = sorted(range(len(moveable_box)), key=lambda i: (moveable_box[i,-1], moveable_box[i,1]), reverse=True)
      moveable_box = moveable_box[sorted_idx]
      for box in moveable_box:
        xyxy = box[:4]
        datum = np.array([(xyxy[0]+xyxy[2])/2, xyxy[1]], np.float32).reshape(1, 2)  # top center
        xy_bars_avail = xy_bars[mask_bars]  # body image consider unused bar_item
        idx_map = np.argwhere(mask_bars).reshape(-1)
        cls_name = idx2unit[int(box[-2])]
        find_bar = False
        if len(xy_bars_avail):
          dis = scipy.spatial.distance.cdist(datum, xy_bars_avail)[0]
          # print(f"cls={cls_name},id={box[-4]},box dis=", dis)  # DEBUG: distance
          idx = np.argmin(dis)
          # print(idx_map, idx, mask_bars, dis[idx])
          if dis[idx] < DIS_BAR_AND_BODY_THRE:
            i = idx_map[idx]
            bar_item = no_body_items[i]
            assert bar_item.body is None
            # if bar_item.body is not None:  # Near with tower-bar, skip
            #   print(f"Warning: (time={self.time}) {idx2unit[int(box[-2])]}(id={box[-4]}) near with tower-bar, which has body(id={bar_item.body[-4]}), skip this (maybe wrong detection)")
            #   continue
            #   # print(i, idx, mask_bars, idx_map)
            # assert bar_item.body is None
            bar_item.body = box
            mask_bars[i] = False
            find_bar = True
        if not find_bar:
          no_bar_box.append(box)
    for box in no_bar_box:
      self._add_bar_item(body=box)
  
  def _update_bar_items_history(self):
    if not np.isinf(self.time):
      while not self.bar_history.empty() and self.time - self.bar_history.queue[0][1] > self.persist:
        id, _ = self.bar_history.get()
        # check latest time
        if (id in self.bar2xywht) and (self.time - self.bar2xywht[id][-1] > self.persist):
          self.bar2xywht.pop(id)
          if id in self.bel_memory:
            self.bel_memory.pop(id)
          if id in self.cls_memory:
            self.cls_memory.pop(id)
    for bar in self.bar_items:
      bar.update_state_bar_info()
  
  def _update_cls_memory(self):
    for item in self.bar_items:
      if item.body is None: continue
      cls = int(item.body[-2])
      most_cls = Counter()
      for b in item.bars:
        if b is None: continue
        id = int(b[-4])
        most_cls.update(self.cls_memory[id])
        self.cls_memory[id].update({cls: 1})
      if len(most_cls):
        most_cls = most_cls.most_common(1)[0][0]
        # if id == 119:
        #   print("id=119, most_cls", most_cls)
        if cls != most_cls:
          print(f"Warning(state): (time={self.time}) bars and body (id={item.body[-4]}) don't have same class, bar class={idx2unit[int(most_cls)]}, body class={idx2unit[int(cls)]}")
          item.body[-2] = most_cls

  def update(self, info: dict, deploy_cards: set):
    """
    Args:
      info (dict): The return in `VisualFusion.process()`,
        which has keys=[time, arena, cards, elixir]
      deploy_cards (set): Get deploy_cards from action_builder.
    """
    self.time: int = info['time'] if not np.isinf(info['time']) else self.time
    self.arena: CRResults = info['arena']
    self.cards: List[str] = info['cards']
    self.elixir: int = info['elixir']
    self.card2idx: dict = info['card2idx']
    self.parts_pos: np.ndarray = info['parts_pos']  # shape=(3, 4), part1,2,3, (x,y,w,h)
    self.box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    self.img = self.arena.get_rgb()
    self.frame_count += 1
    # assert box.shape[-1] == 8, 
    ### Step 0: Update belong memory ###
    self._update_bel_memory()
    ### Step 1: Find text information ###
    self._find_text_info(deploy_cards)
    ### Step 2: Build bar items ###
    self._build_bar_items()
    ### Step 3: Combine units and bar2 with their BarItem ###
    self._combine_bar_items()
    ### Step 4: Update bar history ###
    self._update_bar_items_history()
    ### Step 5: Update class memory, if body exists ###
    self._update_cls_memory()
  
  def get_state(self, verbose=False):
    state = {}
    # state['arena'] = self.img  # NOT NEED IT
    state['time'] = self.time
    state['unit_infos'] = []
    for i in self.bar_items:
      info = i.get_unit_info(verbose=verbose)
      if info is not None:
        state['unit_infos'].append(info)
    state['cards'] = [self.card2idx[c] for c in self.cards]  # next, card1~4
    state['elixir'] = self.elixir
    return state
  
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
    self.body, self.bar2xywht, self.bar_history = body, state.bar2xywht, state.bar_history
    self.bel_memory, self.cls_memory, self.time = state.bel_memory, state.cls_memory, state.time
    self.img = state.img
    self.bars = [self.bar_level, self.bar1, self.bar2]
    self.bel, self.bel_cnt = None, 0
    for i, bar in enumerate(self.bars + [body]):
      if bar is not None and i != 2:  # Don't consider bar2 belong
        bel = int(bar[-1])
        cnt = max(self.bel_memory[int(bar[-4])].values())
        if self.bel is not None:
          if self.bel != bel:
            print(f"Warning(state): (time={self.time}) bars (id={bar[-4]}) don't have same belong")
            if cnt > self.bel_cnt:
              self.bel = bel
              self.bel_cnt = cnt
        else:
          self.bel = bel
          self.bel_cnt = cnt
    if self.bel is None:
      print(f"Warning(state): Only build BarItem by bar2 (id={int(self.bar2[-4])})")
    # assert self.bel is not None, "The belong of BarItem is None"
  
  def debug(self, info):
    cls_name = idx2unit[info['cls']] if (info is not None and info['cls'] is not None) else None
    print(f"Bar Items (cls={cls_name}) (id): ", end='')
    for name in ('bar_level', 'bar1', 'bar2', 'body'):
      box = getattr(self, name)
      if box is not None:
        print(f"{name}={box[-4]}", end=' ')
    print()
    return info
  
  def update_state_bar_info(self):
    if self.body is None: return
    xyxy = self.body[:4]
    cls = self.body[-2]
    xy_body, wh_body = xyxy[:2], xyxy[2:] - xyxy[:2]
    for bar in self.bars:
      if bar is None: continue
      xy_bar = bar[:2]
      id = int(bar[-4])
      # if id in self.bar2xywht:
      #   last_cls = self.bar2xywht[id][-2]
      #   if last_cls != cls:
      #     warnings.warn(colorstr('Warning')+f"Last class is {idx2unit[int(last_cls)]} but new class is {idx2unit[int(cls)]} at time {self.time}")
      self.bar2xywht[id] = np.concatenate([xy_body - xy_bar, wh_body, [self.time]]).astype(np.float32)  # Time maybe np.inf
      self.bar_history.put((id, self.time))
  
  def get_unit_info(self, verbose=False):
    """
    Parse unit_info from bars and body image.

    Args:
      img: The arena image (part2).
    Returns:
      info: The unit info same as `BASE_UNIT_INFO` struct.
    """
    last_time = -1
    info = BASE_UNIT_INFO.copy()
    most_cls = Counter()
    for bar in self.bars:
      if bar is not None: most_cls.update(self.cls_memory[int(bar[-4])])
    if len(most_cls): info['cls'] = most_cls.most_common(1)[0][0]
    if self.body is None:
      for bar in self.bars:
        if bar is None: continue
        id = int(bar[-4]) 
        if id in self.bar2xywht:
          xywht = self.bar2xywht[id]
          if xywht[-1] <= last_time: continue
          last_time = xywht[-1]
          xyxy_box = bar[:4]
          xy = xywht[:2] + xyxy_box[:2]
          info['xy'] = pixel2cell(xy+xywht[2:4]/2)  # [xy[0]+xywht[2]/2, xy[1]+xywht[3]/2]
          xyxy = np.concatenate([xy, xy+xywht[2:4]], 0)
          # info['body'] = extract_img(self.img, xyxy)  # last body size
        else:
          xy = self.center
          xy[1] += BAR_CENTER2BODY_DELTA_Y
          info['xy'] = pixel2cell(xy)
          if info['xy'][1] >= 32: return None  # WRONG detection of bar
    else:
      xyxy = self.body[:4]
      info['xy'] = pixel2cell(xyxy.reshape(2,2).mean(0))
      if info['cls'] is None:  # Only body
        info['cls'] = int(self.body[-2])
      # if info['cls'] != int(self.body[-2]):
      #   warnings.warn(colorstr("Warning")+f"(time={self.time}) bars and body (id={self.body[-4]}) don't have same class")
      # info['body'] = extract_img(self.img, xyxy)
      # info['body'] = xyxy
      bel = int(self.body[-1])
      if bel != self.bel:
        print(f"Warning(state): (time={self.time}) bars and body (id={self.body[-4]}) don't have same belong")
        counter = self.bel_memory[int(self.body[-4])]
        if len(counter):
          cnt = max(counter.values())
          if cnt > self.bel_cnt:
            self.bel_cnt = cnt
            self.bel = bel
    for name in ['bar1', 'bar2']:
      bar = getattr(self, name)
      if bar is None: continue
      xyxy = bar[:4]
      cls_name = idx2unit[int(bar[-2])]
      if cls_name == 'king-tower-bar': xyxy = xyxy2sub(xyxy, SUB_XYXY_KING_TOWER_BAR)
      if cls_name == 'tower-bar': xyxy = xyxy2sub(xyxy, SUB_XYXY_TOWER_BAR[int(bar[-1])])
      info[name] = extract_img(self.img, xyxy)
      # info[name] = xyxy
      # if cls_name in ['king-tower-bar', 'tower-bar']:  # DEBUG
      #   import cv2
      #   cv2.imshow('img', info[name][...,::-1])
      #   cv2.waitKey(0)
    info['bel'] = self.bel
    if self.bel is None: return None  # Only bar2 if useless
    if verbose:
      self.debug(info)
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
