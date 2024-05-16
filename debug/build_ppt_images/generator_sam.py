""" SAM Process Single Image for exapmle"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.build_dataset.constant import path_logs, image_size_part2
from katacr.constants.label_list import idx2unit
from katacr.constants.state_list import idx2state
from tqdm import tqdm
from katacr.utils import Stopwatch

CHECKPOINT_PATH = r"/home/yy/Coding/models/sam_vit_h_4b8939.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cxcywh2xyxy(box, img):
  if box.ndim == 1: box = box.reshape(1, 4)
  h, w = img.shape[:2]
  box[:,[0,2]] *= w
  box[:,[1,3]] *= h
  x1 = box[:,0] - box[:,2] / 2
  y1 = box[:,1] - box[:,3] / 2
  x2 = box[:,0] + box[:,2] / 2
  y2 = box[:,1] + box[:,3] / 2
  box = np.stack([x1, y1, x2, y2], -1).round()
  return box.astype(np.float32)

class Segment:
  def __init__(self):
    self.path_manager = PathManager()
    sam = sam_model_registry['vit_h'](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    self.predictor = SamPredictor(sam)
    print("Loading SAM successfully!!!")

    self.path_log_save = path_logs / "segment_unit"  # The temporary files
    self.path_log_save.mkdir(exist_ok=True)
    self.path_save = self.path_manager.path / "images/segment"  # The dataset files (picked)
    self.path_save.mkdir(exist_ok=True)
    self.cache_idx = {}  # The cache of the index of class in dataset
  
  def _segment(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Args:
      x (np.ndarray): Input image. [shape=(H,W,3)]
      box (np.ndarray): Bounding boxes prompts for image segment. [shape=(N,4), VOC format (xyxy with pixel)]
    Return:
      mask (np.ndarray): The mask with bounding box prompt. [shape=(H,W)]
    """
    if box.ndim == 1:
      box = box.reshape(-1, 4)
    self.predictor.set_image(x)
    box = torch.tensor(box, device=self.predictor.device)
    box = self.predictor.transform.apply_boxes_torch(box, x.shape[:2])
    masks, _, _ = self.predictor.predict_torch(
      point_coords=None,
      point_labels=None,
      boxes=box,
      multimask_output=False
    )
    masks = masks.cpu().numpy()  # shape=(N,1,H,W)
    mask = np.zeros(masks.shape[1:], np.bool_)
    for i in range(masks.shape[0]): 
      mask |= masks[i]
    return mask[0]
  
  def process(self, part_suffix=2, image_name=None, episode=None):
    """
    Process the images in part_id with video_name and episode.
    """
    paths = [Path(image_name).with_suffix('.txt')]
    sw = Stopwatch()
    bar = tqdm(paths)
    for path_box in bar:
      vn, episode, frame = path_box.parts[-3:]  # video_name, episode, frame
      path_img = path_box.parent / (path_box.name.rsplit('.',1)[0] + '.jpg')
      img = np.array(Image.open(str(path_img)).convert('RGB').resize(image_size_part2))
      box = np.loadtxt(str(path_box))
      for b in box:
        sw.__enter__()
        cls = idx2unit[int(b[0])]
        xyxy = cxcywh2xyxy(b[1:5], img)
        states = b[5:].astype(np.int32)
        name = cls
        for i, s in enumerate(states):
          if i == 0: name += '_' + idx2state[s]
          elif s > 0: name += '_' + idx2state[i*10+s]
        path_log_save = self._get_log_save_path(vn, cls, name, suffix='png')
        mask = self._segment(img, xyxy)
        xyxy = xyxy[0].astype(np.int32)
        x = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
        Image.fromarray(x).save(path_log_save.with_stem(path_log_save.stem + '_org'))
        mask = mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]][...,None]
        alpha = (mask * 255).astype(np.uint8)
        x = np.concatenate([x * mask, alpha], axis=-1, dtype=np.uint8)
        Image.fromarray(x).save(path_log_save)
        sw.__exit__()
        bar.set_description(f"{sw.avg_per_s:.2f}box/s,{sw.avg_dt:.2f}s/box")
  
  def _get_log_save_path(self, video_name: str, cls: str, name: str, suffix='jpg'):
    """
    Append the minimal index of `cls` class in dataset to `name`, and
    return the path of the current unit segment to save.
    """
    path_save = self.path_save / cls
    idx = -1  # label index - 1
    if cls in self.cache_idx:
      idx = self.cache_idx[cls]
    elif path_save.exists():
      for path in path_save.glob('*'):
        idx = max(idx, int(path.name.rsplit('.',1)[0].split('_')[-1]))
    idx += 1
    self.cache_idx[cls] = idx
    self.path_log_save_cls = self.path_log_save / f"{video_name}/{cls}"
    self.path_log_save_cls.mkdir(parents=True, exist_ok=True)
    save_path = self.path_log_save_cls / f"{name}_{idx:07}.{suffix}"
    if save_path.exists():
      print(f"The path {save_path} exists, can you rename the {self.path_log_save} and try again?")
      exit()
    return save_path
  
  def background(self):
    paths = self.path_manager.search(subset='images', part=2, video_name='background', regex=r"background\d+.jpg")
    for path_img in tqdm(paths):
      path_box = path_img.parent / (path_img.name.rsplit('.',1)[0] + '.txt')
      img = np.array(Image.open(str(path_img)).convert('RGB'))
      box = np.loadtxt(str(path_box))
      xyxy = cxcywh2xyxy(box[:,1:5], img)
      mask = self._segment(img, xyxy)
      img = img * (1 - mask[...,None]).astype(np.uint8)
      path_log_save = self.path_log_save / "background"
      path_log_save.mkdir(exist_ok=True)
      Image.fromarray(img).save(str(path_log_save / path_img.name))

if __name__ == '__main__':
  segment = Segment()
  segment.process(image_name="/home/yy/Coding/datasets/Clash-Royale-Dataset/images/part2/OYASSU_20210528_episodes/5/03195.jpg")

