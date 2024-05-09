"""
Build action from VisualFusion result
Action data format:
- xy (ndarray | None): range in (36, 64)
- card_id (int): range in (0 ~ 4)
If no action in frame, xy=None, card_id=0; if multi action in intervel frames,
the action will be stacked in Queue and get one action in each require frame.

ActionBuilder.update should use before StateBuilder.update since this 
will update info['cards'] to correct state (solve empty cards).

Update action in 10fps, get_action 5fps.
"""

from katacr.yolov8.custom_result import CRResults
from queue import Queue
import numpy as np
from katacr.constants.label_list import unit2idx
from katacr.policy.perceptron.utils import pixel2cell, LOW_ALPHA, edit_distance
from katacr.ocr_text.paddle_ocr import OCR
from katacr.classification.elixir.predict import ElixirClassifier
from katacr.constants.card_list import card2elixir
import cv2
from pathlib import Path
# path_root = Path(__file__).parents[3]
# path_wrong_classify_images = path_root / "logs/wrong_classify"
# path_wrong_classify_images.mkdir(exist_ok=True)

OCR_TEXT_SIZE = (200, 120)  # bottom center = elixir top center
WRONG_CARD_FRAME_DELTA = 10  # 10 * 0.1 = 1 sec
EMPTY_CARD_UPDATE_FRAME_DELTA = 5  # 5 * 0.1 = 0.5 sec, delay to use last classification result after card empty
EDIT_DISTANCE_THRE = 2  # Levenshtein distance between ocr text **in** target text
ELIXIR_MUTATION_FRAME_DELTA = 3

class ActionBuilder:
  def __init__(self, persist: int=2, ocr: OCR = None):
    """
    Args:
      persist (int): The maximum time to memory in elixir_history (second)
      ocr (OCR): paddle ocr model, if exists.
    Variables:
      elixir_history (Queue): val=(id, time), memory appeared elixir
      elixirs (set): used elixir ids
      card_memory (List): Avaiable cards name, [next_card, card1, card2, card3, card4]
      actions (Queue[Dict]): Action stack in intervel frames
    """
    self.persist = persist
    self.ocr = OCR(lang='en') if ocr is None else ocr
    self.elixir_classifier = ElixirClassifier()
    self.wrong_img_count = 0
    self.last_elixir_num = None
    self.reset()
  
  def reset(self):
    self.elixir_history = Queue()
    self.elixirs = set()
    self.cards_memory = [None] * 5
    self.actions = Queue()
    self.time = 0
    self.last_wrong_card_frame = [None] * 5
    self.frame_count = 0
    self.last_empty_frame = [0] * 5
    self.elixir_mutation_frames = []  # memory last elixir mutation frame, make offset to current action, to make sure action time is correctly
    # self.elixir_count = 0
  
  def _update_elixir(self):
    while not self.elixir_history.empty() and self.time - self.elixir_history.queue[0][1] > self.persist:
      id, _ = self.elixir_history.get()
      if id in self.elixirs:
        self.elixirs.remove(id)
  
  def _update_cards(self):
    self.deploy_cards = set()
    self.cards_memory[0] = self.cards[0]  # next card should correct
    from collections import Counter
    counter = Counter(self.cards)
    common = counter.most_common(1)[0]
    if not (common[1] <= 1 or common[0] == 'empty'):
      print(f"Warning(action): wrong detection, two same card {common[0]} in {self.cards}, skip it")
      return
      # cv2.imshow("wrong", self.img[...,::-1])
      # cv2.waitKey(0)
    # print(f"Action(Time={self.time},frame={self.frame_count}): before {self.cards=}, {self.cards_memory=}, {self.deploy_cards=}")  # DEBUG
    for i, (nc, mc) in enumerate(zip(self.cards, self.cards_memory)):
      wrong_name = False
      if nc != 'empty':
        if mc in ['empty', None]:
          self.cards_memory[i] = nc
        else:  # mc has class name
          if nc != mc:
            duplic = False  # If one card covers another card
            for j in range(5):
              if i != j and self.cards_memory[j] == nc:
                duplic = True
            if not duplic:
              if self.frame_count - self.last_empty_frame[i] <= EMPTY_CARD_UPDATE_FRAME_DELTA:
                self.cards_memory[i] = nc
              else:
                wrong_name = True
                if self.last_wrong_card_frame[i] is not None:
                  if self.frame_count - self.last_wrong_card_frame[i] >= WRONG_CARD_FRAME_DELTA:
                    # print(f"Warning(action): There is a missing action before, since detect_cards={self.cards} and cards_memory={self.cards_memory}, change to detection card")
                    # self.cards_memory[i] = nc
                    print(f"Warning(action): There is a missing action before, since detect_cards={self.cards} and cards_memory={self.cards_memory}")
                    self.cards_memory[i] = nc
                    # assert nc == mc, f"There is a missing action before, since detect_cards={self.cards} and cards_memory={self.cards_memory}"
                else:
                  self.last_wrong_card_frame[i] = self.frame_count
          else:
            if nc in self.deploy_cards:  # drag out and drag back
              self.deploy_cards.remove(nc)
      if nc == 'empty':
        self.last_empty_frame[i] = self.frame_count
        if mc not in ['empty', None]:
          self.deploy_cards.add(mc)
          self.cards[i] = mc  # update cards state
      if not wrong_name:
        self.last_wrong_card_frame[i] = None
    # print(f"Action(Time={self.time},frame={self.frame_count}): {self.cards=}, {self.cards_memory=}, {self.deploy_cards=}")  # DEBUG
  
  def _add_action(self, elixir_box, card_name):
    b = elixir_box
    xyxy = b[:4]
    # find action at elixir bottom center
    cell_xy = pixel2cell(np.array([xyxy[[0,2]].mean(), xyxy[3]], np.int32))
    cell_xy[1] -= 0.2  # down bias 0.2 cell
    card_id = self.cards_memory.index(card_name)
    offset = self.frame_count - (self.elixir_mutation_frames[0] if len(self.elixir_mutation_frames) else self.frame_count)
    if self.frame_count % 2 == 1: offset += 1
    self.elixir_mutation_frames = []
    self.actions.put({'xy': cell_xy, 'card_id': card_id, 'offset': offset})
  
  def _ocr_text(self, elixir_box):
    xyxy = elixir_box[:4]
    # elixir top center
    xy = np.array([xyxy[[0,2]].mean(), xyxy[1]], np.int32)
    size = OCR_TEXT_SIZE
    img_size = self.img.shape[:2][::-1]
    text_xyxy = np.array([xy[0]-size[0]/2, xy[1]-size[1], xy[0]+size[0]/2, xy[1]], np.int32)
    text_xyxy[[0,2]] = text_xyxy[[0,2]].clip(0, img_size[0])
    text_xyxy[[1,3]] = text_xyxy[[1,3]].clip(0, img_size[1])
    text_img = self.img[text_xyxy[1]:text_xyxy[3], text_xyxy[0]:text_xyxy[2]]
    # import cv2  # DEBUG
    # cv2.imshow('text_img', text_img[...,::-1])
    # cv2.waitKey(0)
    results = self.ocr(text_img)
    recs = []
    has_text = False
    for result in results:
      if result is None: continue
      has_text = True
      for info in result:
        det, rec = info
        rec = ''.join([c for c in rec[0].lower() if c in LOW_ALPHA])
        if len(rec) == 0: continue
        recs.append(rec)
        # print(self.deploy_cards)
        for name in self.deploy_cards:
          tname = name.lower().replace('-', '')
          # if rec in name.lower().replace('-', ''):
          if edit_distance(rec, tname, dis='s1') <= EDIT_DISTANCE_THRE:
            return name
    print(f"Warning(action): (time={self.time}) Don't find any {recs} in display_cards: {self.deploy_cards} by elixir (id={elixir_box[-4]})")
    return has_text  # Maybe ocr detection is wrong or other text cover on it, return has_text for further judgment
  
  def _classify_elixir_number(self, elixir_box):
    xyxy = elixir_box[:4].astype(np.int32)
    img = self.img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    pred = self.elixir_classifier(img)
    # print("elixir number:", pred)
    # cv2.imshow("elixir image", img[...,::-1])
    # cv2.waitKey(0)
    # img = cv2.resize(img, (32, 32))
    # Build elixir classifier dataset
    # cv2.imwrite(f"/home/yy/Coding/GitHub/KataCR/logs/offline/elixir/{self.elixir_count}.jpg", img[...,::-1])
    # self.elixir_count += 1
    return pred
  
  def _update_mutation_elixir_num(self):
    if len(self.elixir_mutation_frames):
      for i in range(len(self.elixir_mutation_frames)):
        if self.frame_count - self.elixir_mutation_frames[i] <= ELIXIR_MUTATION_FRAME_DELTA:
          break
      if i > 0:
        self.elixir_mutation_frames = self.elixir_mutation_frames[i:]
    if self.last_elixir_num is not None:
      if self.elixir is None or self.last_elixir_num > self.elixir:
        self.elixir_mutation_frames.append(self.frame_count - 1)  # mutation pre-frame
    self.last_elixir_num = self.elixir
    # if len(self.elixir_mutation_frames):
    #   print(f"Mutation: (frame={self.frame_count}) mutation_frame={self.elixir_mutation_frames}")
    
  def _find_action(self):
    elixir = self.box[self.box[:,-2] == unit2idx['elixir']]
    # print("Elixir box:", elixir)
    if not len(elixir): return
    for b in elixir:
      id = int(b[-4])
      if id in self.elixirs: continue
      elixir_num = self._classify_elixir_number(b)
      if elixir_num >= 0: continue  # 0: noisy, 1: elixir-collector, 0.5, 1: elixir-golem
      if not len(self.deploy_cards):
        print(f"Warning(action): (time={self.time}) No deploy card for elixirs (id={elixir[:,-4]})")
        continue
      self.elixirs.add(id)
      self.elixir_history.put((id, self.time))
      # if len(self.deploy_cards) == 1:
      #   self._add_action(b, self.deploy_cards.pop())
      # else:
      class_name = self._ocr_text(b)
      if class_name == False: continue
      if class_name == True:  # and len(self.deploy_cards) == 1:
        avail_cards = [c for c in self.deploy_cards if card2elixir[c] == -elixir_num]
        if len(avail_cards) == 0:
          print(f"Warning(action): (time={self.time}) There is text but no correct elixir({id=},{elixir_num=}) cost in deploy_cards={self.deploy_cards}, skip it")
          continue
        class_name = avail_cards[0]
        # class_name = next(iter(self.deploy_cards))
      self._add_action(b, class_name)
      self.deploy_cards.remove(class_name)
  
  def get_action(self, verbose=False):
    action = {'xy': None, 'card_id': 0, 'offset': 0}
    if not self.actions.empty():
      action = self.actions.get()
      self.cards_memory[action['card_id']] = 'empty'
    if verbose and action['card_id']:
      print(f"Time: {self.time}, action={action}")
    return action
  
  def update(self, info):
    """
    Args:
      info (dict): The return in `VisualFusion.process()`,
        which has keys=[time, arena, cards, elixir]
    """
    self.time: int = info['time'] if not np.isinf(info['time']) else self.time
    self.arena: CRResults = info['arena']
    self.cards: dict = info['cards']
    self.elixir: int = info['elixir']
    self.card2idx: dict = info['card2idx']
    self.box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    self.img = self.arena.get_rgb()
    self.frame_count += 1
    ### Step 1: Update elixir history ###
    self._update_elixir()
    # print("Elixirs:", self.elixirs)
    ### Step 2: Update card memory ###
    self._update_cards()
    ### Step 3: Update last elixir ###
    self._update_mutation_elixir_num()
    ### Step 4: Find new action ###
    self._find_action()
    # print(f"{info['cards']=}, {self.cards_memory=}")
