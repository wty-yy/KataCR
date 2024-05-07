import lzma
import cv2
from io import BytesIO
import numpy as np
import time

import scipy.spatial
from katacr.utils import Stopwatch
from katacr.utils.detection import build_label2colors
from PIL import Image, ImageDraw, ImageFont
from katacr.policy.perceptron.utils import background_size
from katacr.utils.detection import FONT_PATH
from katacr.constants.label_list import idx2unit
import scipy

DISPLAY_SCALE = 1

class GridDrawer:
  def __init__(self, r=32*DISPLAY_SCALE, c=18*DISPLAY_SCALE, size=background_size):
    self.r, self.c, self.size = r, c, size
    cell_w = self.size[0] // c
    cell_h = self.size[1] // r
    self.cell = np.array([cell_w, cell_h])
    image = self.image = Image.new('RGB', size, color='black')
    draw = self.draw = ImageDraw.Draw(image)
    for x in list(range(0, size[0], cell_w)) + [size[0]-1]:
      draw.line((x, 0, x, size[1]), fill='white', width=1)
    for y in list(range(0, size[1], cell_h)) + [size[1]-1]:
      draw.line((0, y, size[0], y), fill='white', width=1)
    self.used = np.zeros((r, c), np.bool_)
    self.center = np.swapaxes(np.array(np.meshgrid(np.arange(r), np.arange(c))), 0, -1) + 0.5  # (r, c, 2)
  
  def paint(self, xy, color, text=None, fontsize=14, rect=True, circle=False, text_pos='left top', text_color=(255,255,255)):
    cell, draw = self.cell, self.draw
    xy = np.array(xy) * cell
    xyxy = ((int(xy[0])+1, int(xy[1])+1), (int(xy[0]+cell[0])-1, int(xy[1]+cell[1])-1))
    if rect:
      draw.rectangle(xyxy, color)
    if circle:
      # import PIL  # draw right down
      # pil_version = int(PIL.__version__.split('.')[0])
      # w_text, h_text = font.getbbox('0')[-2:] if pil_version >= 10 else font.getsize('0')
      # xyxy = ((xyxy[0][0]+w_text, xyxy[0][1]+h_text-3), xyxy[1])
      draw.ellipse(xyxy, color)
    font = ImageFont.truetype(FONT_PATH, fontsize)
    import PIL  # draw right down
    pil_version = int(PIL.__version__.split('.')[0])
    w_text, h_text = font.getbbox('0')[-2:] if pil_version >= 10 else font.getsize('0')
    text = str(text)
    if text is not None and text_pos == 'left top':
      w_offset = 4 if rect else 6
      for i, text in enumerate(text.split('\n')):
        draw.text((xyxy[0][0]+w_offset, xyxy[0][1]-2+i*h_text), text, text_color, font)
    if text is not None and text_pos == 'right down':
      draw.text((xyxy[0][0]+1, xyxy[0][1]-2), str(text), text_color, font)
  
  def find_near_pos(self, xy):
    yx = np.array(xy)[::-1]
    y, x = yx.astype(np.int32)
    if self.used[y, x]:
      avail_center = self.center[~self.used]
      map_index = np.argwhere(~self.used)
      dis = scipy.spatial.distance.cdist(yx.reshape(1, 2), avail_center)
      y, x = map_index[np.argmin(dis)]
    self.used[y, x] = True
    return x, y

class DataDisplayer:
  def __init__(self, path_data):
    self.path_data = path_data
    sw = Stopwatch()
    with sw:
      bytes = lzma.open(self.path_data).read()
      self.data = np.load(BytesIO(bytes), allow_pickle=True).item()
    print("Load data time used:", sw.dt)
    self.open_windows = set()
  
  def check_new_window(self, name, size, rate=3):
    if name not in self.open_windows:
      self.open_windows.add(name)
      cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      cv2.resizeWindow(name, (np.array(size)*rate).astype(np.int32))

  def display(self):
    unit_cls = list({unit['cls'] for d in self.data['state'] for unit in d['unit_infos'] if unit['cls'] is not None})
    print(f"num unit cls: {len(unit_cls)},", unit_cls)
    label2color = build_label2colors(unit_cls)
    label2color[None] = (255, 255, 255)
    print(label2color)
    for i, d in enumerate(self.data['state']):
      print(f"Time: {d['time']}, cards: {d['cards']}, elixir: {d['elixir']}, unit_num: {len(d['unit_infos'])}")
      arena = GridDrawer()
      for unit in d['unit_infos']:
        for k, v in unit.items():
          if v is not None:
            if isinstance(v, np.ndarray) and v.dtype == np.uint8:
              self.check_new_window(k, v.shape[:2][::-1], rate=3)
              cv2.imshow(k, v[...,::-1])
              if 'bar' in k:
                cv2.imshow(k+'_resize', cv2.resize(v[...,::-1], (24, 8)))
              # cv2.waitKey(0)
            else:
              print(k, v, end=' ')
              if k == 'xy':
                pos = arena.find_near_pos(v*DISPLAY_SCALE)
                print('->', pos, end=' ')
                arena.paint(pos, label2color[unit['cls']], unit['bel'])
        self.check_new_window('arena', arena.image.size, rate=1.5)
        print()
      action = self.data['action'][i]
      if action['card_id']:
        xy = np.array(action['xy'], np.int32)
        arena.paint(xy, (255,236,158), action['card_id'], rect=False, circle=True, text_color=(0,0,0))
      cv2.imshow('arena', np.array(arena.image)[...,::-1])
      cv2.waitKey(0)

if __name__ == '__main__':
  path_data = "/home/yy/Coding/GitHub/KataCR/logs/offline/2024.05.07 22:44:40/golem_ai_1_two_action.npy.xz"
  displayer = DataDisplayer(path_data=path_data)
  displayer.display()
  # drawer = GridDrawer()
  # print(drawer.find_near_pos((8.93777498,28.14355225)))
  # print(drawer.find_near_pos((1, 1.5)))
  # drawer.paint((5, 0), (255, 0, 0), 0)
  # drawer.paint((0, 0), (255, 0, 0), 1)
  # # drawer.paint((0, 5), (255, 0, 0))
  # drawer.paint((10, 3), (0, 0, 255), 0)
  # drawer.image.show()
