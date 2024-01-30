import cv2
from pathlib import Path
from typing import Sequence
from katacr.detection.cfg import image_shape
import glob, os, numpy as np
from tqdm import tqdm

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
    print(files)
  
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
          print("GG")
          raise StopIteration
        path = self.files[self.count]
        self._new_video(path)
        flag, img = self.cap.read()
      self.frame += 1
      s = f"video {self.count+1}/{self.n} ({self.frame}/{self.total_frame}) {path}:"
    
    else:
      self.count += 1
      img = cv2.imread(path)
      s = f"image {self.count}/{self.n} {path}:"
    
    img = cv2.resize(img, image_shape[:2][::-1])
    img = img[::-1]
    img = np.ascontiguousarray(img)
    print(self.count, s)

    return path, img, self.cap, s

if __name__ == '__main__':
  # p = "/home/yy/Coding/GitHub/KataCR/logs/videos.txt"
  p = "/home/yy/Videos/OYASSU_20210528_h264.mp4"
  ds = ImageAndVideoLoader(p)
  p = []
  # bar = tqdm(ds)
  print(len(ds))
  for path, img, cap, s in ds:
    p.append(p)
    # print(path, img.shape, cap, s)
    # bar.set_description(f"{s}")