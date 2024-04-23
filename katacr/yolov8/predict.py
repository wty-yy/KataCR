import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from katacr.yolov8.train import YOLO_CR
"""
Useage:
cd /your/path/KataCR
python katacv/yolov5/detect.py --path detection_files.txt   # detection_files, each line each file path
                                      /your/path/image.jpg  # image formats
                                      /your/path/video.mp4  # video formats
"""
import sys, cv2, glob, os, argparse, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from typing import Sequence
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from PIL import Image
from katacr.utils import Stopwatch, second2str
from katacr.build_dataset.utils.split_part import process_part

IMG_FORMATS = ['jpeg', 'jpg', 'png', 'webp']
VID_FORMATS = ['avi', 'gif', 'm4v', 'mkv' ,'mp4', 'mpeg', 'mpg', 'wmv']

class ImageAndVideoLoader:
  def __init__(self, path: str | Sequence, video_interval=1, cvt_part2=True):
    self.video_interval = video_interval
    self.cvt_part2 = cvt_part2
    if isinstance(path, str) and Path(path).suffix == '.txt':
      path = [p for p in Path(path).read_text().splitlines() if p[0] != '#']
    files = []
    for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
      p = str(Path(p).resolve())
      if '*' in str(p):
        files.extend(sorted(glob.glob(p, recursive=True)))  # recursive
      elif os.path.isdir(p):
        files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # folder
      elif os.path.isfile(p):
        files.append(p)  # file
      else:
        raise FileNotFoundError(f"{p} does not exists!")
    
    imgs = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    vids = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
    ni, nv = len(imgs), len(vids)
    self.n = ni + nv
    self.files = imgs + vids
    self.video_flag = [False] * ni + [True] * nv
    self.mode = 'image'
    if len(vids):
      self._new_video(vids[0])
    else:
      self.cap = None
  
  def _new_video(self, path):
    self.frame = 0
    self.cap = cv2.VideoCapture(path)
    self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.video_interval
  
  def __len__(self):
    return self.n
  
  def __iter__(self):
    self.count = 0  # count for processed file number
    return self
  
  def __next__(self):
    if self.count == self.n:
      raise StopIteration
    path = self.files[self.count]

    if self.video_flag[self.count]:
      self.mode = 'video'
      # flag, img = self.cap.read()
      for _ in range(self.video_interval):
        flag = self.cap.grab()
      flag, img = self.cap.retrieve()
      while not flag:
        self.count += 1
        self.cap.release()
        if self.count >= self.n:
          raise StopIteration
        path = self.files[self.count]
        self._new_video(path)
        for _ in range(self.video_interval):
          flag = self.cap.grab()
        flag, img = self.cap.retrieve()
      img = img[...,::-1]
      self.frame += 1
      s = f"video {self.count+1}/{self.n} ({self.frame}/{self.total_frame}) {path}:"
    
    else:
      self.count += 1
      img = np.array(Image.open(path).convert("RGB"))
      s = f"image {self.count}/{self.n} {path}:"
    
    if self.cvt_part2:
      img = process_part(img, 2)  # check whether should split part2
    # img = img[None,...]
    img = np.ascontiguousarray(img[...,::-1])

    return path, img, self.cap, s
  
class Infer:
  def __init__(self, path_model=None, iou_thre=0.4, conf_thre=0.5, **kwargs):
    self.iou_thre, self.conf_thre = iou_thre, conf_thre
    self.model = YOLO_CR(str(path_model))
    
  def __call__(self, x):
    return self.model.predict(x, iou=self.iou_thre, conf=self.conf_thre)[0]
  
def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=str, default="",
    help="The path of processed file.")
  parser.add_argument("--path-model", type=str, default=None,
    help="The checkpoint directory of the model")
  parser.add_argument("--iou-thre", type=float, default=0.5,
    help="The threshold of iou in NMS")
  parser.add_argument("--conf-thre", type=float, default=0.1,
    help="The threshold of confidence in NMS")
  parser.add_argument("--show-conf", default=False, const=True, nargs='?')
  args = parser.parse_args(input_args)
  p = Path(args.path_model)
  assert p.exists(), "Must give yolo model path"
  args.model_name = p.stem
  return args

path_root = Path(__file__).parents[2]
from katacr.interact.utils import image_show, multiprocessing
def process(args):
  path_source = str(args.source)
  is_file = path_source.rsplit('.', 1)[-1] in (['txt'] + IMG_FORMATS + VID_FORMATS)
  assert is_file, f"Only support this file: {['txt'] + IMG_FORMATS + VID_FORMATS}"
  save_dir = Path(__file__).parents[2] / f"logs/detection" / args.model_name
  save_dir.mkdir(exist_ok=True, parents=True)
  vid_writer, vid_path = [None], [None]
  ds = ImageAndVideoLoader(path_source)
  infer = Infer(**vars(args))
  plot_kwargs = {'pil': True, 'font_size': None, 'conf': args.show_conf, 'line_width': None, 'font': str(path_root / 'utils/fonts/Arial.ttf')}
  show_img_queue = multiprocessing.Queue()
  show_process = None

  sw = Stopwatch()
  for p, x, cap, s in ds:  # path, image, capture, verbose string
    with sw:
      pred = infer(x)
    if ds.mode in ['image', 'video']:
      img = pred.plot(**plot_kwargs)
      save_path = str(save_dir / (f"{Path(p).parent.name}_ep{Path(p).name}"))
      if ds.mode == 'image':
        cv2.imwrite(save_path, img)
      else:  # video
        if vid_path != save_path:  # new video
          vid_path = save_path
          if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
          if cap:  # video
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = img.shape[:2]
          save_path = str(Path(save_path).with_suffix('.mp4'))
          vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(img)
      if show_process is None:
        show_process = multiprocessing.Process(target=image_show, args=(show_img_queue, img.shape[1], img.shape[0]))
        show_process.start()
      show_img_queue.put(img)
      print(f"{s} {sw.dt * 1e3:.1f}ms", end='')
      if ds.mode == 'video':
        second = (ds.total_frame - ds.frame) * sw.dt
        print(f", time left: {second2str(second)}", end='')
      print()
  cv2.waitKey(0)

if __name__ == '__main__':
  args = parse_args("\
--source /home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/OYASSU_20230203_episodes/2.mp4 \
--show-conf \
--path-model /home/yy/Coding/GitHub/KataCR/logs/yolov8_overfit_train/overfit_train_best_20240324_1545.pt\
".split(' '))
  process(args)