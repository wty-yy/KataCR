"""
open video:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import tkinter as tk
import numpy as np
import math
from typing import List
from PIL import Image, ImageTk
from tkinter.font import Font
import multiprocessing, cv2
GUI_MAX_HEIGHT = 1000
gui_configs = {
  'rect': {
    'info': "选框框（最多画26个）",
    'obj': {
      'rectangle': [chr(ord('A')+i) for i in range(26)],
    },
  },
  'dot': {
    'info': "打点点（最多画26个）",
    'obj': {
      'circle': [chr(ord('A')+i) for i in range(26)],
    },
  }
}

def get_colors(n, use_hex=False):
  import matplotlib.pyplot as plt
  cmap = plt.cm.hsv
  step = cmap.N // n
  colors = cmap([i for i in range(0, cmap.N, step)])
  colors = (colors[:, :3] * 255).astype(int)
  if use_hex:
    ret = [''.join(['#'] + [str(hex(x))[-2:].replace('x', '0').upper() for x in c]) for c in colors]
  else:
    ret = [tuple(color) for color in colors]
  return ret

DISTANCE_THRESHOLD = 8
# COLORS = ["#59D5E0", "#F5DD61", "#FAA300", "#F4538A", "#6420AA", "#A5DD9B"]
COLORS = get_colors(26, use_hex=True)

def stream_show(queue: multiprocessing.Queue, stream_id=0, show=True, name='Stream', scale_ratio=0.5):
  open_window = False
  cap = cv2.VideoCapture(stream_id)
  while True:
    flag, img = cap.read()
    if not flag: break
    h, w, _ = img.shape
    while queue.qsize() > 1: queue.get()
    queue.put(img)
    if not open_window and show:
      open_window = False
      cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
      cv2.resizeWindow(name, int(w*scale_ratio), int(h*scale_ratio))
    if show:
      cv2.imshow(name, img)
      cv2.waitKey(1)  # 1ms

class APP:
  def __init__(self):
    self.master = tk.Tk()
    self.TKFONT = Font(family="song ti", size=16)
    self.screen_queue, self.stream_id = multiprocessing.Queue(), 2
    self.screen_process = multiprocessing.Process(target=stream_show, args=(self.screen_queue, self.stream_id, False))
    self.screen_process.daemon = True
    self.screen_process.start()
    img = self.get_screen(pil=True)
    W, H = img.size
    self.scale = GUI_MAX_HEIGHT / H
    self.rate = round(H / W * 100) / 100
    self.W, self.H = int(W * self.scale), int(H * self.scale)
    img = img.resize((self.W, self.H))
    self.canvas = tk.Canvas(self.master, width=self.W, height=self.H)
    self.canvas.pack(side='left', expand=True)
    self.photo = ImageTk.PhotoImage(img)
    self.background = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    self.last_close = None
    self.circles: List[Circle] = []
    self.rectangles: List[Rectangle] = []
    self.functions = {
      'Motion': {
        '0': self._update_label
      },
      'Button-1': {}
    }
    self.cfg_id = 0
    self.cfg_list = list(gui_configs.keys())
    self.info_text = None
    self._update_cfg()
    self._update_bind()
    self._update_img()
    self._update_button()
    self.count = 0
    self.setting = {}
  
  def get_screen(self, pil=True):
    img = self.screen_queue.get()
    if self.screen_queue.qsize() == 0: self.screen_queue.put(img)
    if pil:
      img = Image.fromarray(img[...,::-1])
    self.img = img
    return img
  
  def _update_img(self):
    img = self.get_screen(pil=True).resize((self.W, self.H))
    self.photo = ImageTk.PhotoImage(img)
    self.canvas.itemconfig(self.background, image=self.photo)
    self.background_update_id = self.master.after(1, self._update_img)
  
  def _update_cfg(self):
    self.cfg = gui_configs[self.cfg_list[self.cfg_id]]
    if 'circle' in self.cfg['obj']:
      self.functions['Button-1']['0'] = self._add_circle
      self.obj_info = self.cfg['obj']['circle']
    elif 'rectangle' in self.cfg['obj']:
      self.functions['Button-1']['0'] = self._add_rectangle
      self.obj_info = self.cfg['obj']['rectangle']
    if self.info_text is not None:
      self.info_text.config(text=self.cfg['info'])
  
  def _save_info(self):
    cfg_name = self.cfg_list[self.cfg_id]
    pos = []
    for u in self.circles + self.rectangles:
      pos.append([int(x / self.scale) for x in u.get_position()])
    self.setting[cfg_name] = pos
  
  def _previous_cfg(self):
    self._save_info()
    self.clean()
    if self.cfg_id > 0:
      self.cfg_id -= 1
    self._update_cfg()
  
  def _next_cfg(self):
    # if self.count != len(self.obj_info):
    #   self.info_text.config(text=self.cfg['info']+"\n**必须完成全部框选才能进入下一步**")
    #   return
    self._save_info()
    self.clean()
    if self.cfg_id < len(self.cfg_list) - 1:
      self.cfg_id += 1
      self._update_cfg()
    elif self.cfg_id == len(self.cfg_list) - 1:
      self.master.destroy()
  
  def _update_button(self):
    self.info_text = tk.Label(self.master, text=self.cfg['info'], font=self.TKFONT)
    self.info_text.pack(side='top')

    button_frame = tk.Frame(self.master)
    button_frame.pack(side="bottom")

    button1 = tk.Button(button_frame, text="Previous", font=self.TKFONT, command=self._previous_cfg)
    button1.pack(side="left")

    self.db_btn = DoubleButton(button_frame, "Freeze", "Continue", lambda: self.master.after_cancel(self.background_update_id), self._update_img, self.TKFONT)

    button2 = tk.Button(button_frame, text="Next", font=self.TKFONT, command=self._next_cfg)
    button2.pack(side="left")

    button2 = tk.Button(button_frame, text="More function", font=self.TKFONT, command=self._your_function)
    button2.pack(side="left")
  
  def _your_function(self):
    ...
  
  def _update_bind(self):
    for name, fns in self.functions.items():
      def process(e, fns=fns):
        for fn in fns.values():
          fn(e)
      self.canvas.bind(f"<{name}>", process)
  
  def _update_label(self, event):
    for c in self.circles:
      c._update_label()
    for r in self.rectangles:
      r._update_label()
  
  def _add_label(self):
    color = COLORS[self.count]
    img = ImageTk.PhotoImage(Image.new("RGB", (20, 20), color))
    label = tk.Label(self.master, image=img, compound='left', font=self.TKFONT)
    label.photo = img
    label.pack(anchor='w')
    return label, color
  
  def _add_circle(self, event):
    x, y = event.x, event.y
    if self._find_closest(x, y, dis_thre=DISTANCE_THRESHOLD) is None:
      if self.count == len(self.obj_info): return
      label, color = self._add_label()
      circle = Circle(self, x, y, name=f"{self.obj_info[self.count]}", color=color, label=label, w=30, h=30)
      self.circles.append(circle)
      self.count += 1
    
  def _add_rectangle(self, event):
    x, y = event.x, event.y
    if self._find_closest(x, y, dis_thre=DISTANCE_THRESHOLD) is None:
      if self.count == len(self.obj_info): return
      label, color = self._add_label()
      rect = Rectangle(self, x, y, self.functions, name=f"{self.obj_info[self.count]}", color=color, label=label)
      self.rectangles.append(rect)
      self.count += 1
  
  def _check_nearby(self, event):
    if self.last_close is not None:
      self.canvas.itemconfig(self.last_close, fill='black')
      self.last_close = None
    x, y = event.x, event.y
    ci = self._find_closest(x, y, dis_thre=DISTANCE_THRESHOLD)
    if ci is not None:
      self.canvas.itemconfig(ci, fill='red')
      self.last_close = ci
  
  def _find_closest(self, x, y, dis_thre=float('inf')):
    cs = self.canvas.find_closest(x, y)
    if not len(cs): return None
    cx, cy = self._get_coord(cs[0])
    dis = math.sqrt((x-cx)**2 + (y-cy)**2)
    if dis <= dis_thre:
      return cs[0]
    return None

  def start(self):
    self.master.mainloop()
  
  def _get_coord(self, id, pos='center'):
    xyxy = self.canvas.coords(id)
    if len(xyxy) == 4 and pos == 'center':
      return (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
    return xyxy
  
  def clean(self):
    for c in self.circles: c.remove()
    for r in self.rectangles: r.remove()
    self.circles, self.rectangles = [], []
    self.count = 0

class DoubleButton:
  def __init__(self, master, text1, text2, fn1, fn2, font):
    self.count = 0
    self.text1, self.text2, self.fn1, self.fn2 = text1, text2, fn1, fn2
    self.btn = tk.Button(master, text=text1, command=self.step, font=font)
    self.btn.pack(side='left')
  
  def step(self):
    if self.count % 2 == 0:
      self.fn1()
      self.btn.config(text=self.text2)
    else:
      self.fn2()
      self.btn.config(text=self.text1)
    self.count += 1

class Circle:
  def __init__(
      self, app: APP,
      x, y,  # center of circle
      w=10, h=10, name="Circle", color='black',
      dynamic_color=True, movable=True, bounded=True, drag_bind_fn=None,
      label: tk.Label=None
    ):
    self.app, self.canvas, self.name, self.color, self.bounded, self.drag_bind_fn, self.label = app, app.canvas, name, color, bounded, drag_bind_fn, label
    self.x, self.y, self.w, self.h = x, y, w, h
    self.bound_range = (app.canvas.winfo_width(), app.canvas.winfo_height())
    self.id = self.canvas.create_oval(x-w/2,y-h/2,x+w/2,y+h/2, fill=self.color)
    if dynamic_color:
      self.canvas.tag_bind(self.id, "<Enter>", lambda _: self.canvas.itemconfig(tk.CURRENT, fill='red'))
      self.canvas.tag_bind(self.id, "<Leave>", lambda _: self.canvas.itemconfig(tk.CURRENT, fill=self.color))
    if movable:
      self.canvas.tag_bind(self.id, "<B1-Motion>", self._dragging)
  
  def _update_position(self, nx, ny):
    if self.bounded:
      nx = np.clip(nx, self.w/2, self.bound_range[0]-self.w/2)
      ny = np.clip(ny, self.h/2, self.bound_range[1]-self.h/2)
    self.canvas.move(self.id, nx - self.x, ny - self.y)
    self.x, self.y = nx, ny
  
  def _dragging(self, event):
    self._update_position(event.x, event.y)
    if self.drag_bind_fn is not None:
      self.drag_bind_fn()
  
  def __repr__(self) -> str:
    return f"{self.name}: ({self.x:.0f}, {self.y:.0f})"

  def _update_label(self):
    if self.label is None: return
    self.label.config(text="{:<30}".format(str(self)))
  
  def remove(self):
    self.canvas.delete(self.id)
    if self.label is not None:
      self.label.destroy()
  
  def get_position(self):
    return [self.x, self.y]

class Rectangle:
  def __init__(self, app: APP, x, y, functions: dict, color='black', width=3, name="Rectangle", label: tk.Label = None):
    self.x0, self.y0, self.x1, self.y1 = x, y, None, None
    self.app, self.canvas, self.color, self.width, self.name, self.label = app, app.canvas, color, width, name, label
    self.functions = functions
    self.circls: List[Circle] = []
    self.id = None
    self.origin_functions = {
      'Button-1': functions['Button-1'].copy(),
      'Motion': functions['Motion'].copy()
    }
    self.functions['Button-1']['0'] = self._drag_stop
    self.functions['Motion'][name] = self._dragging
    self.drawing = True
  
  def _drag_stop(self, event):
    self.functions.update(self.origin_functions)
    self.app._update_bind()
    self._update()
    self.circls.append(Circle(self.app, self.x0, self.y0, color=self.color, drag_bind_fn=self._update))
    self.circls.append(Circle(self.app, self.x0, self.y1, color=self.color, drag_bind_fn=self._update))
    self.circls.append(Circle(self.app, self.x1, self.y0, color=self.color, drag_bind_fn=self._update))
    self.circls.append(Circle(self.app, self.x1, self.y1, color=self.color, drag_bind_fn=self._update))
    self.drawing = False
  
  def _dragging(self, event):
    self.x1, self.y1 = event.x, event.y
    self._update()
  
  def _update_label(self):
    if self.label is None or self.x1 is None: return
    self.label.config(text="{:<30}".format(str(self)))
  
  def _update(self):
    if not self.drawing:
      if self.x0 > self.x1: self.x0, self.x1 = self.x1, self.x0
      if self.y0 > self.y1: self.y0, self.y1 = self.y1, self.y0
      if len(self.circls):
        # sort by (x, y) increasely
        old_pos = [(self.x0, self.y0), (self.x0, self.y1), (self.x1, self.y0), (self.x1, self.y1)]
        # Find move circle id, if it exists
        move_id, move_c = 0, None
        for i, (p, c) in enumerate(zip(old_pos, self.circls)):
          if p != (c.x, c.y): move_id = i; move_c = c
        # Get opposite circle id
        opp_id = 3 - move_id
        xyxy = [item for i in [move_id, opp_id] for item in (self.circls[i].x, self.circls[i].y)]
        self.x0, self.y0, self.x1, self.y1 = min(xyxy[0],xyxy[2]), min(xyxy[1],xyxy[3]), max(xyxy[0],xyxy[2]), max(xyxy[1],xyxy[3])
        now_pos = [(self.x0, self.y0), (self.x0, self.y1), (self.x1, self.y0), (self.x1, self.y1)]
        # Swap move cricle to now position
        for i, p in enumerate(now_pos):
          if move_c is not None and p == (move_c.x, move_c.y):
            self.circls[i], self.circls[move_id] = self.circls[move_id], self.circls[i]
            break
        # Update other cricles position
        for p, c in zip(now_pos, self.circls):
          c._update_position(*p)
        # self.circls = sorted(self.circls, key=lambda c: (c.x, c.y))
    if self.id is not None: self.canvas.delete(self.id)
    self.id = self.canvas.create_rectangle(self.x0, self.y0, self.x1, self.y1, outline=self.color, width=self.width)
    for c in self.circls:
      self.canvas.tag_raise(c.id)
  
  def get_position(self):
    return [self.x0, self.y0, self.x1, self.y1]
  
  def __repr__(self) -> str:
    if self.x1 is None: return ""
    self._update()
    return f"{self.name}: ({self.x0:.0f}, {self.y0:.0f}, {self.x1:.0f}, {self.y1:.0f})"
  
  def remove(self):
    for c in self.circls: c.remove()
    self.canvas.delete(self.id)
    if self.label is not None:
      self.label.destroy()

if __name__ == '__main__':
  app = APP()
  app.start()
  print(app.setting)
  import yaml
  print(yaml.dump(app.setting))
