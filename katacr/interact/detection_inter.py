import multiprocessing, cv2
from katacr.interact.utils import image_show
from katacr.detection.detect import Infer, ImageAndVideoLoader, IMG_FORMATS, VID_FORMATS, parse_args
from katacr.utils import Stopwatch, second2str
from pathlib import Path
import numpy as np
from katacr.utils.detection.data import show_box
import time

class Displayer:
  def __init__(self):
    self.img_queue = multiprocessing.Queue()
    self.process = multiprocessing.Process(target=image_show, args=(self.img_queue,(568*3+80)/1.2,(896)/1.2))
    self.process.start()
    self.args = parse_args(f"\
--path /home/yy/Coding/GitHub/KataCR/logs/detection_files.txt \
--path-model /home/yy/Coding/GitHub/KataCR/logs/YOLOv5_v0.5_unit40_s0-checkpoints/YOLOv5_v0.5_unit40_s0-0150 \
--conf-thre 0.6\
".split(' '))

  def __call__(self):
    args = self.args
    path = str(args.path)
    is_file = path.rsplit('.', 1)[-1] in (['txt'] + VID_FORMATS)
    assert is_file, f"Only support this file: {['txt'] + VID_FORMATS}"
    ds = ImageAndVideoLoader(path)
    infer = Infer(**vars(args))

    sw = Stopwatch()
    for p, x, cap, s in ds:  # path, image, capture, verbose string
      start_time = time.time()
      with sw:
        box = infer(x)[0].copy()
      x = cv2.resize(x[0], (int(568/2.5), int(896/2.5)))
      box[:,:4] /= 2.5
      if ds.mode in ['image', 'video']:
        img = np.array(show_box(x, box, verbose=False, use_overlay=True, show_conf=True, fontsize=8))
        box_img = show_box(np.zeros_like(x), box, verbose=False, use_overlay=False, show_conf=True, fontsize=8)
        pad = np.zeros((x.shape[0], 10, 3))
        img = np.concatenate([x, pad, img, pad, box_img], 1).astype(np.uint8)
        self.img_queue.put(img[...,::-1])
      print(f"{s} {sw.dt * 1e3:.1f}ms", end='')
      if ds.mode == 'video':
        second = (ds.total_frame - ds.frame) * sw.dt
        print(f", time left: {second2str(second)}", end='')
      print()
      # time.sleep(max(1/30-time.time()+start_time, 0))  # 30fps

if __name__ == '__main__':
  displayer = Displayer()
  displayer()
