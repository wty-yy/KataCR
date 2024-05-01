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

class CardClassifier:
  def __init__(self, weight_path=weight_path, load_step=None):
    ckpt_mngr = CheckpointManager(weight_path)
    if load_step is None:
      load_step = int(sorted(Path(weight_path).glob('*'))[-1].name)
    load_info = ckpt_mngr.restore(load_step)
    variables, cfg = load_info['variables'], load_info['config']
    self.img_size = cfg['image_size']
    self.idx2card = cfg['idx2card']
    self.card2idx = cfg['card2idx']
    # print(self.idx2card)
    model_cfg = ModelConfig(**cfg)
    train_cfg = TrainConfig(**cfg)
    self.model = ResNet(cfg=model_cfg)
    self.model.create_fns()
    state = self.model.get_states(train_cfg, train=False)
    self.state = state.replace(params=variables['params'], tx=None, opt_state=None, batch_stats=variables['batch_stats'])
    dummy = np.zeros((*train_cfg.image_size[::-1], 3), np.uint8)
    self.__call__(dummy)
  
  def __call__(self, x, keepdim=False, cvt_label=True, verbose=False):
    """
    Args:
      x (np.ndarray): RGB img.
      keepdim: If taggled and img number is 1, return pred[0].
      cvt_label: If taggled, convert predicted index to label name.
    """
    if not isinstance(x, list): x = [x]
    l = []
    for img in x:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      if img.shape[:2][::-1] != self.img_size:
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
      img = img[..., None]
      l.append(img)
    x = np.array(l)
    if x.dtype == np.uint8: x = x.astype(np.float32) / 255.
    logits = jax.device_get(self.model.predict(self.state, x))
    pred = np.argmax(logits, -1)
    if cvt_label:
      cards = []
      for i in pred: cards.append(self.idx2card[str(i)])
      pred = cards
    if verbose:
      print("class:", logits[0][pred[0]], "conf:", pred)
      cv2.imshow('img', x[0,...,::-1])
      cv2.waitKey(0)
    if x.shape[0] == 1 and not keepdim:
      return pred[0]
    return pred
  
  def process_part3(self, x: np.ndarray, pil=False, cvt_label=True, verbose=False):
    """
    Args:
      x (np.ndarray): The part3 split by katacr/build_dataset/utils/split_part.py process_part(),
        only accept one image (x.ndim==3).
      pil (bool): If taggled, the image `x` is RGB format.
      cvt_label (bool): If taggled, the classification index will be converted to label name.
      verbose (bool): If taggled, each card image will be showed.
    Returns:
      result (List[str]): Detection name for each cards: next card, card1, card2, card3, card4
    """
    from katacr.build_dataset.utils.split_part import extract_bbox
    from katacr.build_dataset.constant import part3_bbox_params
    if not pil: x = x[...,::-1]
    params = part3_bbox_params
    results = []
    for param in params:
      img = extract_bbox(x, *param)  # xywh for next image position
      # cv2.imshow("card", img[...,::-1])
      # cv2.waitKey(0)
      results.append(self(img, cvt_label=cvt_label, verbose=verbose))
    return results

def test_cls():
  for p in Path("/home/yy/Coding/datasets/Clash-Royale-Dataset/images/card_classification/").glob('*'):
    img = cv2.imread(str(p))
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    pred = predictor(img[...,::-1])
    img = cv2.putText(img, pred[0], (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
    cv2.imshow('Predict', img)
    cv2.waitKey(0)

def test_part3():
  from katacr.build_dataset.utils.split_part import process_part
  # root_path = "/home/yy/Coding/GitHub/KataCR/logs/split_image"
  # for p in Path(root_path).glob("part3_*.jpg"):
  p = "/home/yy/Videos/CR_Videos/test/golem_ai/test1.jpg"
  img = process_part(cv2.imread(str(p)), 3)
  pred = predictor.process_part3(img, pil=False)
  print(pred)
  cv2.imshow("pred", img)
  cv2.waitKey(0)

def test_val():
  from train import DatasetBuilder, path_dataset
  ds_builder = DatasetBuilder(str(path_dataset / "images/card_classification"), 0)
  train_cfg = TrainConfig(batch_size=1)
  val_ds = ds_builder.get_dataloader(train_cfg, mode='val')
  for x, y in val_ds:
    if predictor.idx2card[str(int(y[0]))] == 'flying-machine':
      x, y = x.numpy().astype(np.float32) / 255., y.numpy().astype(np.int32)
      print(x.shape)
      pred = predictor(x[0])
      print(pred)
      cv2.imshow('val', x[0,...,::-1])
      cv2.waitKey(0)

if __name__ == '__main__':
  predictor = CardClassifier(weight_path)
  # test_cls()
  # test_val()
  test_part3()
