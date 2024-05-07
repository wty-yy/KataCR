from katacr.yolov8.combo_detect import ComboDetector
from katacr.ocr_text.paddle_ocr import OCR
from katacr.classification.predict import CardClassifier
from pathlib import Path
from katacr.build_dataset.utils.split_part import process_part
import cv2
import numpy as np

path_root = Path(__file__).parents[3]
path_detectors = [
  path_root / './runs/detector1_v0.7.13.pt',
  path_root / './runs/detector2_v0.7.13.pt',
]
path_classifier = path_root / 'logs/CardClassification-checkpoints'

def draw_text(
  img, text, pos=(0, 0), text_color=(0, 255, 0), text_color_bg=(0, 0, 0),
  font_scale=1, font_thickness=2, pos_format='left top',
  font=cv2.FONT_HERSHEY_SIMPLEX):
    # if isinstance(pos, np.ndarray): pos = [int(x) for x in pos]
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size[0] + 5, text_size[1] + 5
    if pos_format == 'left top':  # find left top
      x, y = pos
    elif pos_format == 'left bottom':
      x, y = pos[0], pos[1] - text_h + 5
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

class VisualFusion:
  def __init__(self, ocr_onnx=False, ocr_gpu=True):
    self.ocr = OCR(onnx=ocr_onnx, use_gpu=ocr_gpu, lang='en')
    self.yolo = ComboDetector(path_detectors)
    self.classifier = CardClassifier(path_classifier)
    self.open_window = False
  
  def process(self, x, pil=False):
    """
    Args:
      x (np.ndarray): Input image
      pil (bool): If taggled, image `x` is RGB format.
    Return:
      info (dict): with keys:
        time (int): Time passed. (Part1)
        arena (CRResults inherited from Results): The result is given by YOLOv8. (Part2)
        cards (List[str]): {card_name0, card_name1, card_name2, card_name3, card_name4}. (Part3)
        elixir (int): The elixir we have. (Part3)
    """
    self.parts = parts = []
    parts_pos = []
    for i in range(3):
      img, box_params = process_part(x, i+1, verbose=True)
      parts.append(img)
      parts_pos.append(box_params)
    parts_pos = np.array(parts_pos)
    self.parts_pos = (parts_pos.reshape(-1, 2) * np.array(x.shape[:2][::-1])).astype(np.int32).reshape(-1, 4)
    time = self.ocr.process_part1(parts[0], pil=pil)
    arena = self.yolo.infer(parts[1], pil=pil)
    cards = self.classifier.process_part3(parts[2], pil=pil)
    elixir = self.ocr.process_part3_elixir(parts[2], pil=pil)
    self.info = dict(
      time=time, arena=arena, cards=cards, elixir=elixir,
      card2idx=self.classifier.card2idx, idx2card=self.classifier.idx2card,
      parts_pos=self.parts_pos)
    return self.info
  
  def render(self, x, pil=False, verbose=False):
    self.process(x, pil=pil)
    time, arena, cards, elixir = self.info['time'], self.info['arena'], self.info['cards'], self.info['elixir']
    arena = arena.show_box(show_conf=True)
    parts = self.parts; parts_pos = self.parts_pos
    c = cards
    texts = [f"{time}", f"{c[:2]}\n,{c[2:]},\nelixir: {elixir}"]
    if verbose:
      if not self.open_window:
        cv2.namedWindow("time", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("time", parts[0].shape[:2][::-1])
        cv2.namedWindow("arena", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("arena", arena.shape[:2][::-1])
        cv2.namedWindow("cards", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("cards", parts[2].shape[:2][::-1])
      print(texts[0])
      cv2.imshow("time", parts[0])
      cv2.imshow("arena", arena)
      print(texts[1])
      cv2.imshow("cards", parts[2])
    
    ### Part1 ###
    rx = x.copy()  # render image
    if pil: rx = rx[...,::-1]  # RGB -> BGR

    # rx = cv2.putText(rx, texts[0], parts_pos[0,:2]+np.array([-60,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(144, 233, 205), thickness=2)
    ratio = x.shape[0] / 1280
    draw_text(rx, texts[0], parts_pos[0,:2]+np.array([0,-10]), (144,233,205), (0,0,0), 0.6 * ratio, pos_format='left bottom')
    ### Part2 ###
    xywh = parts_pos[1]; xyxy = parts_pos[1].copy()
    xyxy[2:] += xywh[:2]
    rx[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = cv2.resize(arena, xywh[2:], interpolation=cv2.INTER_CUBIC)
    ### Part3 ###
    for i, line in enumerate(texts[1].split('\n')):
      xywh = parts_pos[2]
      draw_text(rx, line, (xywh[0], int(xywh[1]-50+i*30*ratio)), (253,253,253), (0,0,0), 0.6 * ratio, pos_format='left bottom')
      # rx = cv2.putText(rx, line, (xywh[0], xywh[1]-50+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(253,253,253), thickness=2)
    if verbose:
      if not self.open_window:
        cv2.namedWindow("render", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("render", rx.shape[:2][::-1])
        self.open_window = True
      cv2.imshow("render", rx)
    return rx

if __name__ == '__main__':
  visual = VisualFusion()
  # path_img = "/home/yy/Pictures/ClashRoyale/demos/592x1280/test1.png"
  path_img = "/home/yy/Coding/GitHub/KataCR/logs/offline/2024.04.21 16:41:26/debug_org.jpg"
  img = cv2.imread(path_img)
  visual.render(img, pil=False, verbose=True)
  cv2.waitKey(0)
