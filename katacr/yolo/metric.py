from katacr.utils.related_pkgs.jax_flax_optax_orbax import *

@jax.jit
def logits2prob(logits):  # (..., (x,y,w,h,c,num_classes))
  conf = jax.nn.sigmoid(logits[...,4:5]) * jnp.max(jax.nn.softmax(logits[...,4+1+13:]))
  cls = jnp.argmax(logits[...,4+1+13:], axis=-1, keepdims=True)
  states = logits[...,5:18]
  s1 = (states[...,0:1] >= 0).astype(jnp.float32)
  s2 = jnp.argmax(states[...,1:6], axis=-1, keepdims=True)
  s3 = (states[...,6:7] >= 0).astype(jnp.float32)
  s4 = (states[...,7:8] >= 0).astype(jnp.float32)
  s5 = (states[...,8:9] >= 0).astype(jnp.float32)
  s6 = (states[...,9:10] >= 0).astype(jnp.float32)
  s7 = jnp.argmax(states[...,10:13], axis=-1, keepdims=True)
  states = [s1, s2, s3, s4, s5, s6, s7]
  return jnp.concatenate([logits[...,:4], conf, *states, cls], axis=-1)

def logits2prob_from_list(pred_pixel):
  N = pred_pixel[0].shape[0]
  return jnp.concatenate([logits2prob(pred_pixel[i]).reshape(N, -1, 13) for i in range(3)], axis=1)

from katacr.utils.detection import iou, iou_multiply
@partial(jax.jit, static_argnums=[3,4])
def nms_boxes_and_mask(boxes, iou_threshold=0.3, conf_threshold=0.2, max_num_box=100, iou_format='diou'):
  M = max_num_box
  sort_idxs = jnp.argsort(boxes[:,4])[::-1][:M]  # only consider the first `max_num_box`
  boxes = boxes[sort_idxs]
  ious = iou_multiply(boxes[:,:4], boxes[:,:4], format=iou_format)
  mask = (boxes[:,4] > conf_threshold) & (~jnp.diagonal(jnp.tri(M,k=-1) @ (ious > iou_threshold)).astype('bool'))
  return boxes, mask

def get_pred_bboxes(pred, iou_threshold=0.3, conf_threshold=0.2):
  ret = []
  for i in range(pred.shape[0]):
    bboxes_pred, mask = nms_boxes_and_mask(pred[i], iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    bboxes_pred = bboxes_pred[mask]
    ret.append(bboxes_pred)
  return ret

def mAP(boxes, target_boxes, iou_threshold=0.5):
  """
  Calculate the mAP (AP: area under PR curve) of the boxes and the target_boxes with the iou threshold.
  @params::boxes.shape=(N,6) and last dim is (x,y,w,h,c,cls).
  @params::target_boxes.shape=(N,5) and last dim is (x,y,w,h,cls).
  """
  classes = jnp.unique(target_boxes[:,4])
  APs = 0
  for cls in classes:
    p, r = 1.0, 0.0  # update
    if (boxes[:,5]==cls).sum() == 0: continue
    box1 = boxes[boxes[:,5]==cls]
    sorted_idxs = jnp.argsort(box1[:,4])[::-1]  # use argsort at conf, don't use sort!
    box1 = box1[sorted_idxs]
    box2 = target_boxes[target_boxes[:,4]==cls]
    TP, FP, FN, AP = 0, 0, box2.shape[0], 0
    used = [False for _ in range(box2.shape[0])]
    for i in range(box1.shape[0]):
      match = False
      for j in range(box2.shape[0]):
        if used[j] or iou(box1[i,:4], box2[j,:4])[0] <= iou_threshold: continue
        TP += 1; FN -= 1; used[j] = True; match = True
        break
      if not match: FP += 1
      last_p, p, last_r, r = p, TP/(TP+FP), r, TP/(TP+FN)
      AP += (last_p + p) * (r - last_r) / 2
    APs += AP
  return APs / classes.size

def coco_mAP(boxes, target_boxes):
  """
  Calculate the mAP with iou threshold [0.5,0.55,0.6,...,0.9,0.95]
  """
  ret = 0
  for iou_threshold in 0.5+jnp.arange(10)*0.05:
    ret += mAP(boxes, target_boxes, iou_threshold)
  return ret / 10

def calc_AP50_AP75_AP(pred_bboxes: list, bboxes, num_bboxes):
  AP50, AP75, AP = 0, 0, 0
  mAP_fn = lambda i, thre: mAP(
    boxes=pred_bboxes[i],
    target_boxes=bboxes[i][:num_bboxes[i]],
    iou_threshold=thre
  )
  coco_mAP_fn = lambda i: coco_mAP(
    boxes=pred_bboxes[i],
    target_boxes=bboxes[i][:num_bboxes[i]]
  )
  for i in range(len(pred_bboxes)):
    AP50 += (mAP_fn(i, 0.5) - AP50) / (i + 1)
    AP75 += (mAP_fn(i, 0.75) - AP75) / (i + 1)
    AP += (coco_mAP_fn(i) - AP) / (i + 1)
  return AP50, AP75, AP
