from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.detection import iou_multiply, nms
from katacr.utils.detection.utils_ap import ap_per_class
import numpy as np

class BasePredictor:
  """
  The father class used to predict and metrics for the YOLO model.
  Must define predict method in subclass.

  Args:
    state: YOLO model state.
    pbox: Predicted bounding boxes. List[box.shape=(M,6), elem=(x,y,w,h,conf,cls)]
    tcls: Class of target bounding boxes. List[cls.shape=(M',)]
    tp: Ture positive for the `pbox`. List[tp.shape=(M,len(iout))]
    iout: The threshold of iou for deciding whether is the ture positive. List[int]
  """
  pbox: List[np.ndarray]  # np.float32
  tcls: List[np.ndarray]  # np.int32
  tp: List[np.ndarray]  # np.bool_
  iout: jax.Array

  def __init__(self, state: train_state.TrainState, iout=None, image_shape: Tuple=(896, 576, 3)):
    self.state = state
    self.iout = jnp.linspace(0.5, 0.95, 10) if iout is None else iout
    self.image_shape = image_shape
    if type(self.iout) == float:
      self.iout = jnp.array([self.iout,])
    self.reset()
  
  def reset(self, state: train_state.TrainState = None):
    self.pbox, self.tcls, self.tp = [], [], []
    if state is not None:
      self.state = state
  
  def update(self, x, tbox=None, tnum=None, nms_iou=0.65, nms_conf=0.001):
    """
    Update the prediction variables.

    Args: (`tbox`, `tnum` and `B` can be `None`, \
        just predict the bounding boxes for `x`), \
        `B` is the number of the batch size.
      x: The input of the model. [shape=(B,H,W,C) or (H,W,C)]
      tbox: The target bounding boxes. [shape=(B,M,6), or (M,6)]
      tnum: The number of the target bounding boxes. [shape=(B,) or int]
      nms_iou: The threshold of the iou in NMS.
      nms_conf: The threshold of the confidence in NMS.
    """
    if x.ndim == 3: x = x[None,...]
    if tbox is None and tnum is None:
      pbox, pnum = jax.device_get(self.pred_and_nms(self.state, x, nms_iou, nms_conf))
    else:
      assert(tbox is not None and tnum is not None)
      if tbox.ndim == 2: tbox = tbox[None,...]
      if type(tnum) == int: tnum = jnp.array((tnum,))
      pbox, pnum, tp = jax.device_get(self.pred_and_nms_and_tp(
        self.state, x, nms_iou, nms_conf, tbox, tnum
      ))
    n = x.shape[0]
    for i in range(n):
      self.pbox.append(pbox[i][:pnum[i]])
      if tbox is not None:
        self.tcls.append(tbox[i][:tnum[i],-1].astype(np.int32))
        self.tp.append(tp[i][:pnum[i]])
    return self.pbox[-n:]
  
  def ap_per_class(self):
    """
    Compute average percision (AP) by \
      the area of under recall and precision curve (AUC) for each class.

    Return:
      p: Precision for each class with confidence bigger than 0.1. [shape=(Nc,)]
      r: Recall for each class with confidence bigger than 0.1. [shape=(Nc,)]
      ap: Average precision for each class with different iou thresholds. [shape=(Nc,tp.shape[1])]
      f1: F1 coef for each class with confidence bigger than 0.1. [shape=(Nc,)]
      ucls: Class labels after being uniqued. [shape=(Nc,)]
    """
    pbox = np.concatenate(self.pbox, axis=0)
    return ap_per_class(
      tp=np.concatenate(self.tp, axis=0),
      conf=pbox[:,4],
      pcls=pbox[:,-1],
      tcls=np.concatenate(self.tcls, axis=0)
    )
  
  def save_csv(self, path, p50, r50, ap50, ap75, ap, ucls, idx2name):
    import csv
    with open(path, 'w') as file:
      writer = csv.writer(file)
      writer.writerow(['cls_id', 'cls_name', 'P@50_val', 'R@50_val', 'AP@50_val', 'AP@75_val', 'mAP_val'])
      for i in range(p50.shape[0]):
        writer.writerow([int(ucls[i]), idx2name[int(ucls[i])], float(p50[i]), float(r50[i]), float(ap50[i]), float(ap75[i]), float(ap[i])])
  
  def p_r_ap50_ap75_map(self, path_csv: Path = None, idx2name: dict = None):
    """
    Return:
      p50: Precision with 0.5 iou threshold and bigger than 0.1 confidence.
      r50: Recall with 0.5 iou threshold and bigger than 0.1 confidence.
      ap50: Average precision by AUC with 0.5 iou threshold.
      ap75: Average precision by AUC with 0.75 iou threshold.
      map: Mean average precision by AUC with mean of 10 \
        different iou threshold [0.5:0.05:0.95].
    """
    p, r, ap, f1, ucls = self.ap_per_class()
    p50, r50, ap50, ap75, ap = p[:,0], r[:,0], ap[:,0], ap[:,5], ap.mean(1)
    if path_csv: self.save_csv(path_csv, p50, r50, ap50, ap75, ap, ucls, idx2name)
    p50, r50, ap50, ap75, map = p50.mean(), r50.mean(), ap50.mean(), ap75.mean(), ap.mean()
    return p50, r50, ap50, ap75, map

  def predict(self, state: train_state.TrainState, x: jax.Array):
    """
    Subclass implement.
    Don't forget: `@partial(jax.jit, static_argnums=0)`

    Args:
      state: TrainState `self.state`
      x: Input image. [shape=(N,H,W,3)]
    Return:
      y: All predict box. [shape=(N,num_pbox,7), elem:(x,y,w,h,conf,side,cls)]
    """
    pass

  @partial(jax.jit, static_argnums=[0])
  def pred_bounding_check(self, pbox):
    x1 = jnp.maximum(pbox[...,0] - pbox[...,2] / 2, 0)
    y1 = jnp.maximum(pbox[...,1] - pbox[...,3] / 2, 0)
    x2 = jnp.minimum(pbox[...,0] + pbox[...,2] / 2, self.image_shape[1])
    y2 = jnp.minimum(pbox[...,1] + pbox[...,3] / 2, self.image_shape[0])
    w, h = x2 - x1, y2 - y1
    return jnp.concatenate([jnp.stack([x1+w/2, y1+h/2, w, h], -1), pbox[...,4:]], -1)

  @partial(jax.jit, static_argnums=[0,3,4,5])
  def pred_and_nms(
    self, state: train_state.TrainState, x: jax.Array,
    iou_threshold: float, conf_threshold: float, nms_multi: float = 30
  ):
    pbox = self.predict(state, x)
    pbox = self.pred_bounding_check(pbox)
    pbox, pnum = jax.vmap(
      nms, in_axes=[0, None, None, None], out_axes=0
    )(pbox, iou_threshold, conf_threshold, nms_multi)
    return pbox, pnum
  
  @partial(jax.jit, static_argnums=[0,3,4])
  def pred_and_nms_and_tp(
    self, state: train_state.TrainState, x: jax.Array,
    iou_threshold: float, conf_threshold: float,
    tbox: jax.Array, tnum: jax.Array
  ):
    pbox, pnum = self.pred_and_nms(state, x, iou_threshold, conf_threshold)
    pbox, tp = jax.vmap(self.compute_tp, in_axes=[0,0,0,0,None], out_axes=0)(
      pbox, pnum, tbox, tnum, self.iout
    )
    return pbox, pnum, tp
  
  @staticmethod
  @jax.jit
  def compute_tp(pbox, pnum, tbox, tnum, iout):
    """
    Compute the true positive for each `pbox`. Time complex: O(NM)

    Args:
      pbox: The predicted bounding boxes. [shape=(N,7), elem=(x,y,w,h,conf,side,cls)]
      pnum: The number of available `pbox`. [int]
      tbox: The target bounding boxes. [shape=(M,6), elem=(x,y,w,h,side,cls)]
      tnum: The number of available `tbox`. [int]
      iout: The iou thresholds of deciding true positive. [shape=(1,) or (10,)]
      num_classes: The number of all classes. [int]
    
    Return:
      pbox: The input `pbox` rearrange by increasing confidence.
      tp: The ture positive for the `pbox` after rearrange
    """
    sort_i = jnp.argsort(-pbox[:,4])
    pbox = pbox[sort_i]  # Decrease by confidence
    tp = jnp.zeros((pbox.shape[0],iout.shape[0]), jnp.bool_)
    def solve(tp):  # If tnum > 0
      ious = iou_multiply(pbox[:,:4], tbox[:,:4])  # shape=(N,M)
      bel = jnp.full_like(tp, -1, dtype=jnp.int32)  # calculate belong for each iou threshold
      # Get tp and belong
      def loop_i_fn(i, value):  # i=0,...,pnum-1
        tp, bel = value
        iou = ious[i]
        j = jnp.argmax((tbox[:,-1]==pbox[i,-1]) * iou)  # belong to tbox[j]
        # Round to 0.01, https://github.com/rafaelpadilla/Object-Detection-Metrics
        def update(value):
          tp, bel = value
          tp = tp.at[i].set((iou[j].round(2) >= iout - 1e-5) & (j < tnum))
          bel = bel.at[i].set(j)
          return tp, bel
        tp, bel = jax.lax.cond(tbox[j,-1]==pbox[i,-1], update, lambda x: x, (tp, bel))
        return tp, bel
      tp, bel = jax.lax.fori_loop(0, pnum, loop_i_fn, (tp, bel))
      # Remove duplicate belong
      bel = jnp.where(tp, bel, -1)
      mask = jnp.zeros_like(bel, jnp.bool_)
      def loop_j_fn(j, mask):  # j=0,...,tnum-1
        tmp = bel == j
        tmp = tmp.at[jnp.argmax(tmp,0), jnp.arange(tmp.shape[1])].set(False)  # use argmax to get the first nonzero arg
        mask = mask | tmp
        return mask
      mask = jax.lax.fori_loop(0, tnum, loop_j_fn, mask)
      tp = tp * (~mask)
      return tp
    tp = jax.lax.cond(tnum > 0, solve, lambda x: x, tp)
    return pbox, tp

def load_pred_and_target_file(path_pred: Path, path_tg: Path, format='coco'):
  cls2idx = {'_num': 0}
  def load_file(path: Path):
    box = []
    with open(path, 'r') as file:
      for line in file.readlines():
        if len(line) == 0: continue
        line = line.split()
        if len(line) == 6:
          cls, conf, x, y, w, h = [x if i == 0 else float(x) for i, x in enumerate(line)]
        else:
          cls, x, y, w, h = [x if i == 0 else float(x) for i, x in enumerate(line)]
        if format == 'coco':
          x += w / 2
          y += h / 2
        if cls not in cls2idx:
          cls2idx[cls] = cls2idx['_num']; cls2idx['_num'] += 1
        cls = cls2idx[cls]
        if len(line) == 6:
          box.append(jnp.stack([x, y, w, h, conf, cls]))
        if len(line) == 5:
          box.append(jnp.stack([x, y, w, h, cls]))
    return jnp.stack(box)
  def load_dir(path_dir: Path):
    box, num = [], []
    for path in sorted(path_dir.iterdir()):
      if path.is_file():
        box.append(load_file(path))
        num.append(box[-1].shape[0])
    for i in range(len(box)):
      box[i] = jnp.pad(box[i], ((0, max(num) - box[i].shape[0]), (0, 0)))
    return jnp.stack(box), jnp.stack(num)
  pbox, pnum = load_dir(path_pred)
  tbox, tnum = load_dir(path_tg)
  return pbox, pnum, tbox, tnum, cls2idx

if __name__ == '__main__':
  # path_tg = Path("/home/wty/Coding/GitHub/Object-Detection-Metrics-master/groundtruths")
  # path_pred = Path("/home/wty/Coding/GitHub/Object-Detection-Metrics-master/detections")
  # metric_from_file(path_tg, path_pred)
  pass
