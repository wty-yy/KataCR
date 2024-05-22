"""
Useage:
cd /your/path/KataCR
python katacv/yolov5/detect.py --path detection_files.txt   # detection_files, each line each file path
                                      /your/path/image.jpg  # image formats
                                      /your/path/video.mp4  # video formats
"""
import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # allocate GPU memory as needed
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
  def __init__(self, path: str | Sequence):
    if isinstance(path, str) and Path(path).suffix == '.txt':
      path = Path(path).read_text().split()
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
    self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
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
      flag, img = self.cap.read()
      while not flag:
        self.count += 1
        self.cap.release()
        if self.count == self.n:
          raise StopIteration
        path = self.files[self.count]
        self._new_video(path)
        flag, img = self.cap.read()
      img = img[...,::-1]
      self.frame += 1
      s = f"video {self.count+1}/{self.n} ({self.frame}/{self.total_frame}) {path}:"
    
    else:
      self.count += 1
      img = np.array(Image.open(path).convert("RGB"))
      s = f"image {self.count}/{self.n} {path}:"
    
    img = process_part(img, 2)  # check whether should split part2
    img = img[None,...]
    img = np.ascontiguousarray(img)

    return path, img, self.cap, s
  
class Infer:
  def __init__(self, model_name="YOLOv5_v0.4.3", load_id=80, path_model=None, iou_thre=0.4, conf_thre=0.5, **kwargs):
    self.iou_thre, self.conf_thre = iou_thre, conf_thre
    from katacr.detection.parser import get_args_and_writer
    self.args = get_args_and_writer(no_writer=True, input_args=f"--model-name {model_name} --load-id {load_id} --batch-size 1".split())

    print("Loading model weights...")
    from katacr.detection.model import get_state
    self.state = get_state(self.args)
    
    if path_model is None:
      from katacr.utils.model_weights import load_weights
      self.state = load_weights(self.state, self.args)
    elif os.path.isfile(path_model):
      with open(path_model, 'rb') as file:
        self.state = flax.serialization.from_bytes(self.state, file.read())
      print(f"Succesfully load weights from {path_model}")
    else:
      from katacr.utils.model_weights import load_weights_orbax
      self.state = load_weights_orbax(self.state, path_model)
    self.state = self.state.replace(params=None, batch_stats=None, grads=None)
    
    from katacr.detection.predict import Predictor
    self.predictor = Predictor(self.args, self.state)
  
  @partial(jax.jit, static_argnums=[0])
  def preprocess(self, x):
    if x.ndim == 3:
      x = x[None, ...]
    w = jnp.array([x.shape[i+1] / self.args.image_shape[i] for i in [1, 0]])
    w = jnp.r_[w, w, [1] * 3].reshape(1,1,7)
    x = jnp.array(x, dtype=jnp.float32) / 255.
    x = jax.image.resize(x, (x.shape[0], *self.args.image_shape), method="trilinear")
    pbox, pnum = self.predictor.pred_and_nms(self.state, x, iou_threshold=self.iou_thre, conf_threshold=self.conf_thre, nms_multi=10)
    pbox = pbox * w
    return pbox, pnum
  
  def __call__(self, x):
    pbox, pnum = jax.device_get(self.preprocess(x))
    pbox = [pbox[i][:pnum[i]] for i in range(pbox.shape[0])]
    return pbox
  
def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", type=str, default="",
    help="The path of processed file.")
  parser.add_argument("--model-name", type=str, default="YOLOv5_v0.5",
    help="The name of model in /logs/{model_name}-checkpoints")
  parser.add_argument("--load-id", type=int, default=150,
    help="The id of loaded model")
  parser.add_argument("--path-model", type=str, default=None,
    help="The checkpoint directory of the model")
  parser.add_argument("--iou-thre", type=float, default=0.5,
    help="The threshold of iou in NMS")
  parser.add_argument("--conf-thre", type=float, default=0.1,
    help="The threshold of confidence in NMS")
  return parser.parse_args(input_args)

from katacr.utils.detection.data import show_box
def process(args):
  path = str(args.path)
  is_file = path.rsplit('.', 1)[-1] in (['txt'] + IMG_FORMATS + VID_FORMATS)
  assert is_file, f"Only support this file: {['txt'] + IMG_FORMATS + VID_FORMATS}"
  save_dir = Path(__file__).parents[2] / f"logs/detection" / args.model_name
  save_dir.mkdir(exist_ok=True, parents=True)
  vid_writer, vid_path = [None], [None]
  ds = ImageAndVideoLoader(path)
  infer = Infer(**vars(args))

  sw = Stopwatch()
  for p, x, cap, s in ds:  # path, image, capture, verbose string
    with sw:
      pbox = infer(x)
    for i, box in enumerate(pbox):
      if ds.mode in ['image', 'video']:
        img = show_box(x[i], box, verbose=False, use_overlay=True, show_conf=True)
        save_path = str(save_dir / (f"{Path(p).parent.name}_ep{Path(p).name}"))
        if ds.mode == 'image':
          img.save(save_path)
        else:  # video
          if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
              vid_writer.release()
            if cap:  # video
              fps = cap.get(cv2.CAP_PROP_FPS)
              w, h = img.size
            save_path = str(Path(save_path).with_suffix('.mp4'))
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
          vid_writer.write(np.array(img)[...,::-1])
      print(f"{s} {sw.dt * 1e3:.1f}ms", end='')
      if ds.mode == 'video':
        second = (ds.total_frame - ds.frame) * sw.dt
        print(f", time left: {second2str(second)}", end='')
      print()

if __name__ == '__main__':
  # p = "/home/yy/Coding/GitHub/KataCR/logs/videos.txt"
  # p = "/home/yy/Videos/OYASSU_20210528_h264.mp4"
  # p = "/home/wty/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20210528_episodes/1_h264.mp4"
  # p = "/home/wty/Coding/GitHub/KataCR/logs/detection_files.txt"
  # args = parse_args(f"--path {p}".split())
  args = parse_args()
  process(args)