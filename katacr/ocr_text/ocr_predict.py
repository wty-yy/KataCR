# -*- coding: utf-8 -*-
'''
@File  : ocr_ctc.py
@Time  : 2023/10/14 20:03:24
@Author  : wty-yy
@Version : 1.0
@Blog  : https://wty-yy.xyz/
@Desc  : 
'''
import os, sys
sys.path.append(os.getcwd())

from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.related_pkgs.utility import *
from katacr.utils import Stopwatch

from katacr.ocr_text.crnn_model import TrainState

@partial(jax.jit, static_argnums=2)
def predict(state: TrainState, x, blank_id=0) -> Tuple[jax.Array, jax.Array]:
  logits = state.apply_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    x, train=False
  )
  pred_probs = jax.nn.softmax(logits)
  pred_idxs = jnp.argmax(pred_probs, -1)  # (B,T)
  mask = (
    (pred_idxs != jnp.pad(pred_idxs[:,:-1], ((0,0),(1,0)))) &
    (pred_idxs != blank_id)
  )
  conf = jnp.prod(pred_probs.max(-1), -1)  # (B,) confidence
  return pred_idxs, mask, conf

import numpy as np
def apply_mask(pred_idxs, mask, max_len=23) -> jax.Array:
  y_pred = np.zeros((pred_idxs.shape[0], max_len), dtype=np.int32)
  for i in range(pred_idxs.shape[0]):
    seq = pred_idxs[i][mask[i]]
    N = min(max_len, seq.size)
    y_pred[i,:N] = seq[:N]
  return y_pred

def predict_result(
    state: TrainState,
    x: jax.Array,
    max_len: int,
    idx2ch: dict
  ) -> Sequence[str]:
  pred_idx, mask, conf = jax.device_get(predict(state, x))
  y_pred = apply_mask(pred_idx, mask, max_len)
  pred_seq = []
  for i in range(y_pred.shape[0]):
    seq = []
    for j in range(y_pred.shape[1]):
      if y_pred[i,j] == 0: break
      seq.append(chr(idx2ch[y_pred[i,j]]))
    pred_seq.append("".join(seq))
  return pred_seq, conf

from katacr.ocr_text.parser import OCRArgs
from katacr.ocr_text.parser import get_args_and_writer
from katacr.ocr_text.crnn_model import get_ocr_crnn_state
import cv2
class OCRText:
  def __init__(self, args: OCRArgs = get_args_and_writer(input_args="")):
    self.args = args
    self.state = get_ocr_crnn_state(args)
    weights = ocp.PyTreeCheckpointer().restore(str(args.path_weights))
    self.state = self.state.replace(
      params=weights['params'],
      batch_stats=weights['batch_stats']
    )
    print("OCR text has prepared!")
  
  def predict(self, images: Sequence[np.array]):
    x = np.zeros((
      len(images),
      self.args.image_height,
      self.args.image_width, 1
    ))
    for i, image in enumerate(images):
      image = cv2.resize(image, (self.args.image_width, self.args.image_height))
      if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[...,None]
      if image.ndim == 3 and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)[...,None]
      if image.ndim == 2:
        image = image[...,None]
      x[i] = image
    return predict_result(
      self.state, x, 
      self.args.max_label_length, self.args.idx2ch
    )

if __name__ == '__main__':
  # from katacr.ocr_text.parser import get_args_and_writer
  ocr_text = OCRText()

  path1 = Path("/home/wty/Pictures/test/ocr/2400p_part4_up.jpg")
  from katacr.utils import load_image_array
  image1 = load_image_array(path1)
  # image2 = load_image_array(path2)
  # image3 = load_image_array(path3)
  # from katacr.build_train_dataset.split_frame_parts import process_part4
  # part4 = list(process_part4(image).values())
  # part4 = [image1, image2, image3]
  part4 = [image1]
  ocr_text.predict(part4)  # compile

  sw = Stopwatch()
  with sw:
    print(ocr_text.predict(part4))
  print("OCR used time:", sw.dt)
