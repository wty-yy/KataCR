import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # allocate GPU memory as needed
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from katacr.classification.train import ModelConfig, TrainConfig, ResNet
from katacr.utils.ckpt_manager import CheckpointManager
import numpy as np
import jax, jax.numpy as jnp
import cv2

weight_path = path_root / 'logs/CardClassification-checkpoints'

class Predictor:
  def __init__(self, weight_path, load_step=None):
    ckpt_mngr = CheckpointManager(weight_path)
    if load_step is None:
      load_step = int(sorted(Path(weight_path).glob('*'))[-1].name)
    load_info = ckpt_mngr.restore(load_step)
    variables, cfg = load_info['variables'], load_info['config']
    self.img_size = cfg['image_size']
    self.idx2card = cfg['idx2card']
    model_cfg = ModelConfig(**cfg)
    train_cfg = TrainConfig(**cfg)
    self.model = ResNet(cfg=model_cfg)
    self.model.create_fns()
    state = self.model.get_states(train_cfg, train=False)
    self.state = state.replace(params=variables['params'], tx=None, opt_state=None, batch_stats=variables['batch_stats'])
    dummy = np.zeros((*train_cfg.image_size[::-1], 3), np.uint8)
    self.__call__(dummy)
  
  def __call__(self, x, keepdim=False, cvt_label=True):
    if not isinstance(x, list): x = [x]
    l = []
    for img in x:
      if img.shape[:2][::-1] != self.img_size:
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
      l.append(img)
    x = np.array(l)
    if x.dtype == np.uint8: x = x.astype(np.float32) / 255.
    pred = np.argmax(jax.device_get(self.model.predict(self.state, x)), -1)
    if x.shape[0] == 0 and not keepdim:
      return pred[0]
    if cvt_label:
      cards = []
      for i in pred: cards.append(self.idx2card[str(i)])
      pred = cards
    return pred

if __name__ == '__main__':
  predictor = Predictor(weight_path)
  # img = cv2.imread("/home/yy/Coding/datasets/Clash-Royale-Dataset/images/card_classification/archer-queen.jpg")
  for p in Path("/home/yy/Coding/datasets/Clash-Royale-Dataset/images/card_classification/").glob('*'):
    img = cv2.imread(str(p))
    pred = predictor(img[...,::-1])
    img = cv2.putText(img, pred[0], (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
    cv2.imshow('Predict', img)
    cv2.waitKey(0)
