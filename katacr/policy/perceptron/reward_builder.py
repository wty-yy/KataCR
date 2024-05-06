import numpy as np
from katacr.ocr_text.paddle_ocr import OCR
from katacr.yolov8.custom_result import CRResults
from katacr.constants.label_list import unit2idx
from katacr.policy.perceptron.utils import extract_img, xyxy2center, xyxy2sub, pil_draw_text
from katacr.build_dataset.generation_config import except_king_tower_unit_list
import cv2

TARGET_SIZE_KING_TOWER_BAR = (140, 40)
TARGET_SIZE_TOWER_BAR = (96, 30)
XYXY_KING_TOWER_BAR_NUM = [(0.3, 0.4, 1.0, 1.0), (0.4, 0.0, 1.0, 0.5)]
XYXY_TOWER_BAR_NUM = [(0.26, 0.1, 1.0, 0.9), (0.26, 0.0, 1.0, 0.8)]
XYXY_KING_TOWER = [(185, 700, 385, 885), (195, 0, 385, 200)]
XYXY_TOWER = [[(50, 625, 175, 770), (400, 625, 525, 770)], [(50, 135, 175, 285), (405, 135, 525, 285)]]
OCR_NUM_CONF_THRE = 0.9
DESTROY_FRAME_DELTA_THRE = 10
MAX_DELTA_HP = 1600
ELIXIR_OVER_FRAME = 10  # 0.1 * 5 = 1 sec

class RewardBuilder:
  def __init__(self, ocr: OCR = None):
    self.ocr = OCR(lang='en') if ocr is None else ocr
    self.reset()
  
  def reset(self):
    self.full_hp = {'tower': [1, 1], 'king-tower': [1, 1]}
    self.hp_tower = np.full((2, 2), -1, np.int32)
    self.hp_king_tower = np.full((2,), -1, np.int32)
    self.last_elixir_over_frame = None
    self.last_tower_destroy_frame = np.full((2, 2), -1, np.int32)
    self.last_king_tower_destroy_frame = np.full((2,), -1, np.int32)
    self.frame_count = 0
    self.time = 0
  
  def _ocr_hp(self, xyxy, target_size):
    img = extract_img(self.img, xyxy, target_size=target_size)
    imgs = [img, img[...,::-1]]
    results = self.ocr(imgs, det=False)[0]
    # print(results)  # DEBUG
    # cv2.imshow('img', img[...,::-1])
    # cv2.waitKey(0)
    # results = sorted(results, key=lambda x: x[0][0])  # sort by det's left top point
    nums = []
    for rec, conf in results:
      # print("DEBUG: conf=", conf)  # DEBUG
      rec = rec.lower()
      num = ''.join([c for c in rec.strip() if (c in [str(i) for i in range(10)])])
      if conf < OCR_NUM_CONF_THRE or len(num) == 0:
        print(f"Wrong: (time={self.time}) Don't find bar number (rec={rec}) or conf={conf:.4f} is low")
        num = -1
      else: num = int(num)
      # print(results)  # DEBUG
      # cv2.imshow('img', img[...,::-1])
      # cv2.waitKey(0)
      nums.append(num)
    if np.mean(nums) == nums[0]: return nums[0]
    return -1
  
  def _has_other_item_cover(self, xyxy):
    if not isinstance(xyxy, np.ndarray): xyxy = np.array(xyxy)
    if xyxy.ndim == 1: xyxy = xyxy.reshape(1, 4)
    boxes = []
    for name in ['emote', 'clock', 'elixir']:
      boxes.append(self.box[self.box[:,-2] == unit2idx[name]])
    emote = np.concatenate(boxes, 0)[:,:4]
    a1, a2 = np.array_split(np.expand_dims(xyxy, 1), 2, -1)  # (N, 1, 4), N=1
    b1, b2 = np.array_split(np.expand_dims(emote, 0), 2, -1)  # (1, M, 4)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)  # (N, M)
    return inter.sum() > 0
  
  def render(self, img, reward):
    """
    Write reward at (0, 0) in (BGR) image.
    """
    text = f"Reward: {reward:.4f}" if reward is not None else f"Reward: None"
    text += f"\nFrame: {self.frame_count}"
    img = pil_draw_text(img, (0, 0), text)
    return np.array(img)[...,::-1]
  
  def get_reward(self, verbose=False):
    now_hp_tower = np.full((2, 2), -1, np.int32)
    now_hp_king_tower = np.full((2,), -1, np.int32)
    ### King Tower ###
    get_unit_box = lambda unit: self.box[self.box[:,-2] == unit2idx[unit]]
    king_tower_bar_box = get_unit_box('king-tower-bar')
    # assert len(king_tower_bar_box) <= 2
    if len(king_tower_bar_box):
      for b in king_tower_bar_box:
        bel = int(b[-1])
        if 60 < xyxy2center(b[:4])[1] < 840: continue  # Wrong king tower bar
        if self._has_other_item_cover(b[:4]): continue
        xyxy = xyxy2sub(b[:4], XYXY_KING_TOWER_BAR_NUM[bel])
        hp = self._ocr_hp(xyxy, TARGET_SIZE_KING_TOWER_BAR)
        if self.hp_king_tower[bel] != -1:
          if hp > self.hp_king_tower[bel]:
            print(f"Warning(reward): (time={self.time}) king-tower id={b[-4]}, ocr_hp={hp} > old hp={self.hp_king_tower[bel]}, there maybe wrong detection before")
          delta = self.hp_king_tower[bel] - hp
          if abs(delta) > MAX_DELTA_HP:
            print(f"Warning(reward): (time={self.time}) king-tower id={b[-4]}, old_hp-ocr_hp={delta}'s abs {MAX_DELTA_HP=}, ignore it")
            hp = -1
        now_hp_king_tower[bel] = hp
    # check king-tower is destroied
    king_tower_box = get_unit_box('king-tower')
    frame = self.last_king_tower_destroy_frame
    for i in range(2):
      if sum(king_tower_box[:,-1]==i) or self._has_other_item_cover(XYXY_KING_TOWER[i]):  # tower exists or uncertain
        frame[i] = -1
      elif frame[i] != -1:
        if self.frame_count - frame[i] >= DESTROY_FRAME_DELTA_THRE:
          now_hp_king_tower[i] = 0
      else:
        frame[i] = self.frame_count
    ### Tower ###
    tower_bar_box = get_unit_box('tower-bar')
    for b in tower_bar_box:
      xy = xyxy2center(b[:4])
      size = self.img.shape[:2][::-1]
      i, j = int(xy[1] < size[1] / 2), int(xy[0] > size[0] / 2)
      if self._has_other_item_cover(b[:4]): continue
      xyxy = xyxy2sub(b[:4], XYXY_TOWER_BAR_NUM[int(b[-1])])
      hp = self._ocr_hp(xyxy, TARGET_SIZE_TOWER_BAR)
      if self.hp_tower[i,j] != -1:
        if hp > self.hp_tower[i,j]:
          print(f"Warning(reward): (time={self.time}) tower id={b[-4]}, ocr_hp={hp} > old hp={self.hp_tower[i,j]}, there maybe wrong detection before")
        delta = self.hp_tower[i,j] - hp
        if abs(delta) > MAX_DELTA_HP:
          print(f"Warning(reward): (time={self.time}) tower id={b[-4]}, old_hp-ocr_hp={delta}'s abs > {MAX_DELTA_HP=}, ignore it")
          hp = -1
      now_hp_tower[i,j] = hp
    # check tower is destroied
    tower_box = np.concatenate([get_unit_box(name) for name in except_king_tower_unit_list], 0)
    for b in tower_box:
      xy = xyxy2center(b[:4])
      size = self.img.shape[:2][::-1]
      i, j = int(xy[1] < size[1] / 2), int(xy[0] > size[0] / 2)
      self.last_tower_destroy_frame[i,j] = -2  # tower exist symbol
    frame = self.last_tower_destroy_frame
    for i in range(2):
      for j in range(2):
        if frame[i,j] == -2 or self._has_other_item_cover(XYXY_TOWER[i][j]):  # tower exists or uncertain
          frame[i,j] = -1
        elif frame[i,j] != -1:
          if self.frame_count - frame[i,j] >= DESTROY_FRAME_DELTA_THRE:
            now_hp_tower[i,j] = 0
        else:
          frame[i] = self.frame_count
        # if self._has_emote(XYXY_TOWER[i][j]):  # DEBUG
        #   print("Has emote in tower-patch", i, j)
    # print("Last tower destroy time:", self.last_tower_destroy_time)
    ### Calculate ###
    reward = {'tower': 0, 'king-tower': 0, 'r_': 0, 'elixir': 0}
    if verbose:
      print("OLD hp:", self.hp_tower, self.hp_king_tower)  # DEBUG
      print("NOW hp:", now_hp_tower, now_hp_king_tower)  # DEBUG
    # Tower
    for i, (olds, nows) in enumerate(zip(self.hp_tower, now_hp_tower)):
      flag = (-1) ** (i+1)
      for j, (old, now) in enumerate(zip(olds, nows)):
        full_hp = self.full_hp['tower'][i]
        if old not in [-1, 0] and now == 0:  # destroied: reward +/- 1
          reward['r_'] += flag * 1
        if now != -1:
          if old == -1 and now != 0:  # first view
            self.full_hp['tower'][i] = max(now, full_hp)
          elif old != -1:  # alived: reward +/- (delta/full)
            reward['tower'] += flag * (old - now) / full_hp
          self.hp_tower[i,j] = now
    # King Tower
    for i, (old, now) in enumerate(zip(self.hp_king_tower, now_hp_king_tower)):
      flag = (-1) ** (i+1)
      full_hp =  self.full_hp['king-tower'][i]
      if old not in [-1, 0] and now == 0:  # destroied: reward +/- 3
        reward['r_'] += flag * 3
      if now != -1:
        if old == -1 and now != 0:  # first view
          self.full_hp['king-tower'][i] = max(now, full_hp)
          # print("PROD:", np.prod(self.hp_tower[i]))  # DEBUG
          if np.prod(self.hp_tower[i]) != 0:  # towers all alive and open king-tower:  reward -/+ 0.1
            reward['r_'] += -flag * 0.1
        elif old != -1:  # alived: reward +/- (delta/full)
          reward['king-tower'] += flag * (old - now) / full_hp
        self.hp_king_tower[i] = now
    if verbose:
      print("FULL HP:", self.full_hp)
    # Elixir
    if self.elixir == 10:
      if self.last_elixir_over_frame is not None:
        if self.frame_count - self.last_elixir_over_frame >= ELIXIR_OVER_FRAME:
          reward['elixir'] -= 0.05
          self.last_elixir_over_frame = self.frame_count
      else:
        self.last_elixir_over_frame = self.frame_count
    else:
      self.last_elixir_over_frame = None
    total_reward = sum(reward.values())
    if verbose:
      print(f"Time={self.time}, Frame={self.frame_count}, {reward=}, {total_reward=:.4f}")
    return total_reward
  
  def update(self, info):
    self.time: int = info['time'] if not np.isinf(info['time']) else self.time
    self.arena: CRResults = info['arena']
    self.elixir: int = info['elixir']
    self.img = self.arena.get_rgb()
    self.box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    self.frame_count += 1
