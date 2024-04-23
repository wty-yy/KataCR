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
from katacr.policy.utils import pixel2cell
from katacr.ocr_text.paddle_ocr import OCR

OCR_TEXT_SIZE = (200, 120)  # bottom center = elixir top center
LOW_ALPHA = [chr(ord('a')+i) for i in range(26)]
WRONG_CARD_FRAME_DELTA = 10  # 10 * 0.2 = 2 sec

class ActionBuilder:
  def __init__(self, persist: int=2):
    """
    Args:
      persist (int): The maximum time to memory in elixir_history (second)
    Variables:
      elixir_history (Queue): val=(id, time), memory appeared elixir
      elxiirs (set): used elixir ids
      card_memory (List): Avaiable cards name, [next_card, card1, card2, card3, card4]
      actions (Queue[Dict]): Action stack in intervel frames
    """
    self.persist = persist
    self.ocr = OCR(lang='en')
    self.reset()
  
  def reset(self):
    self.elixir_history = Queue()
    self.elixirs = set()
    self.cards_memory = [None] * 5
    self.actions = Queue()
    self.time = 0
    self.last_wrong_card_frame = [None] * 5
    self.frame_count = 0
  
  def _update_elixir(self):
    while not self.elixir_history.empty() and self.time - self.elixir_history.queue[0][1] > self.persist:
      id, _ = self.elixir_history.get()
      if id in self.elixirs:
        self.elixirs.remove(id)
  
  def _update_cards(self):
    self.deploy_cards = set()
    self.cards_memory[0] = self.cards[0]  # next card should correct
    for i, (nc, mc) in enumerate(zip(self.cards, self.cards_memory)):
      wrong_name = False
      if nc != 'empty':
        if mc in ['empty', None]:
          self.cards_memory[i] = nc
        else:  # mc has class name
          if nc != mc:
            wrong_name = True
            if self.last_wrong_card_frame[i] is not None:
              if self.frame_count - self.last_wrong_card_frame[i] >= WRONG_CARD_FRAME_DELTA:
                assert nc == mc, f"There is a missing action before, since detect_cards={self.cards} and cards_memory={self.cards_memory}"
            else:
              self.last_wrong_card_frame[i] = self.frame_count
          else:
            if nc in self.deploy_cards:  # drag out and drag back
              self.deploy_cards.remove(nc)
      if nc == 'empty' and mc != 'empty':
        self.deploy_cards.add(mc)
      if not wrong_name:
        self.last_wrong_card_frame[i] = None
    print(f"Action(Time={self.time}): {self.cards=}, {self.cards_memory=}, {self.deploy_cards=}")  # DEBUG
  
  def _add_action(self, elixir_box, card_name):
    b = elixir_box
    xyxy = b[:4]
    # find action at elixir bottom center
    cell_xy = pixel2cell(np.array([xyxy[[0,2]].mean(), xyxy[3]], np.int32))
    cell_xy[1] -= 0.2  # down bias 0.2 cell
    card_id = self.cards_memory.index(card_name)
    self.actions.put({'xy': cell_xy, 'card_id': card_id})
    self.cards_memory[card_id] = 'empty'
  
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
        recs.append(rec)
        for name in self.deploy_cards:
          if rec in name.lower().replace('-', ''):
            return name
    print(f"Warning(action): (time={self.time}) Don't find any {recs} in display_cards: {self.deploy_cards} by elixir (id={elixir_box[-4]})")
    return has_text  # Maybe ocr detection is wrong or other text cover on it, return has_text for further judgment
    
  def _find_action(self):
    elixir = self.box[self.box[:,-2] == unit2idx['elixir']]
    # print("Elixir box:", elixir)
    if not len(elixir): return
    for b in elixir:
      id = int(b[-4])
      if id in self.elixirs: continue
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
      if class_name == True and len(self.deploy_cards) == 1:
        class_name = next(iter(self.deploy_cards))
      self._add_action(b, class_name)
      self.deploy_cards.remove(class_name)
  
  def get_action(self, verbose=False):
    action = {'xy': None, 'card_id': 0}
    if not self.actions.empty():
      action = self.actions.get()
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
    ### Step 1: Udpate elixir history ###
    self._update_elixir()
    # print("Elixirs:", self.elixirs)
    ### Step 2: Update card memory ###
    self._update_cards()
    ### Step 3: Find new action ###
    self._find_action()
    # print(f"{info['cards']=}, {self.cards_memory=}")
    info['cards'] = self.cards_memory
    self.frame_count += 1
