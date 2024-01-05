import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from katacr.build_dataset.utils.datapath_manager import PathManager

CHECKPOINT_PATH = r"/home/wty/Coding/models/sam_vit_h_4b8939.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cxcywh2xyxy(box, img):
  h, w = img.shape[:2]
  box[:,[0,2]] *= w
  box[:,[1,3]] *= h
  x1 = box[:,0] - box[:,2] / 2
  y1 = box[:,1] - box[:,3] / 2
  x2 = box[:,0] + box[:,2] / 2
  y2 = box[:,1] + box[:,3] / 2
  box = np.stack([x1, y1, x2, y2], -1).round()
  return box.astype(np.float32)

class SegmentUnit:
  def __init__(self):
    self.path_manager = PathManager()
    sam = sam_model_registry['vit_h'](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    self.predictor = SamPredictor(sam)
  
  def _segment(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Args:
      x (np.ndarray): Input image. [shape=(H,W,3)]
      box (np.ndarray): Bounding boxes prompts for image segment. \
        [shape=(N,4), VOC format (xyxy with pixel)]
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
    ret = np.zeros(masks.shape[1:], np.bool_)
    for i in range(masks.shape[0]):
      ret |= masks[i]
    return ret
  
  def process(self, part_suffix=2, video_name=None, episode=None):
    """
    Process the images in part_id with video_name and episode.
    """
    if episode is not None:
      # TODO
      pass
    paths = self.path_manager.sample(part=part_suffix, video_name=video_name, name=str(episode))
    for path in paths:
      print(path)
      break

if __name__ == '__main__':
  segment_unit = SegmentUnit()
  segment_unit.process()

