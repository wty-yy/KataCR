from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.detection.predictor import BasePredictor
from katacr.detection.loss import cell2pixel
from katacr.detection.parser import YOLOv5Args
from katacr.constants.state_list import num_state_classes
from katacr.detection.train_state import TrainState

class Predictor(BasePredictor):
  
  def __init__(self, args: YOLOv5Args, state: TrainState, iout=None):
    super().__init__(state, iout, args.image_shape)
    self.args = args

  @partial(jax.jit, static_argnums=0)
  def predict(self, state: TrainState, x: jnp.ndarray):
    logits = state.apply_fn(
      {'params': state.ema['params'], 'batch_stats': state.ema['batch_stats']},
      x, train=False
    )
    y, batch_size = [], x.shape[0]
    # elems=(x,y,w,h,conf,*state_classes,*classes)
    for i in range(3):
      xy = (jax.nn.sigmoid(logits[i][...,:2]) - 0.5) * 2.0 + 0.5
      xy = cell2pixel(xy, scale=2**(i+3))
      wh = (jax.nn.sigmoid(logits[i][...,2:4]) * 2) ** 2 * self.args.anchors[i].reshape(1,3,1,1,2)
      conf = jax.nn.sigmoid(logits[i][...,4:5])
      cls = jax.nn.sigmoid(logits[i][...,4+1+num_state_classes:])
      conf = conf * jnp.max(cls, axis=-1, keepdims=True)
      cls = jnp.argmax(cls, axis=-1, keepdims=True)
      states = logits[i][...,5:5+num_state_classes]
      s1 = (states[...,0:1] >= 0).astype(jnp.float32)
      # s2 = jnp.argmax(states[...,1:6], axis=-1, keepdims=True)
      # s3 = (states[...,6:7] >= 0).astype(jnp.float32)
      # s4 = (states[...,7:8] >= 0).astype(jnp.float32)
      # s5 = (states[...,8:9] >= 0).astype(jnp.float32)
      # s6 = (states[...,9:10] >= 0).astype(jnp.float32)
      # s7 = jnp.argmax(states[...,10:13], axis=-1, keepdims=True)
      # states = [s1, s2, s3, s4, s5, s6, s7]
      states = [s1]
      y.append(jnp.concatenate([xy,wh,conf,*states,cls], -1).reshape(batch_size,-1,7))
    y = jnp.concatenate(y, 1)  # shape=(batch_size,all_pbox_num,5+1+1)
    return y
  