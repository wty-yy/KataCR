# -*- coding: utf-8 -*-
'''
@File    : utils_ap.py
@Time    : 2023/12/03 09:31:33
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : 
The utility functions for compute AP@[.5:.05:.95], mAP.
Reference: https://github.com/WongKinYiu/ScaledYOLOv4/blob/yolov4-large/utils/general.py
'''
from katacr.utils.related_pkgs.utility import *
import numpy as np
import matplotlib.pyplot as plt

def ap_per_class(tp, conf, pcls, tcls):
  """
  Compute AP for each class in `np.unique(tcls)`.

  Args:
    tp: True positive of the predicted bounding boxes. [shape=(N,10) or (N,1)]
    conf: Confidence of the predicted bounding boxes. [shape=(N,)]
    pcls: Class label of the predicted bounding boxes. [shape=(N,)]
    tcls: Class label of the target bounding boxes. [shape=(M,)]
  
  Return:
    p: Precision for each class with confidence bigger than 0.1. [shape=(Nc,tp.shape[1])]
    r: Recall for each class with confidence bigger than 0.1. [shape=(Nc,tp.shape[1])]
    ap: Average precision for each class with different iou thresholds. [shape=(Nc,tp.shape[1])]
    f1: F1 coef for each class with confidence bigger than 0.1. [shape=(Nc,)]
    ucls: Class labels after being uniqued. [shape=(Nc,)]
  """
  sort_i = np.argsort(-conf)
  tp, conf, pcls = tp[sort_i], conf[sort_i], pcls[sort_i]
  ucls = np.unique(tcls)
  shape = (len(ucls), tp.shape[1])
  ap, p, r = np.zeros(shape), np.zeros(shape), np.zeros(shape)
  pr_score = 0.1
  for i, cls in enumerate(ucls):
    idx = pcls == cls
    # number of predict and target boxes with class `cls`
    n_p, n_t = idx.sum(), (tcls==cls).sum()
    if n_p == 0: continue
    fpc = (1-tp[idx]).cumsum(0)  # cumulate false precision
    tpc = tp[idx].cumsum(0)  # cumulate true precision
    recall = tpc / n_t
    r[i] = np.interp(-pr_score, -conf[idx], recall[:,0])  # conf[idx] decrease
    precision = tpc / (tpc + fpc)
    p[i] = np.interp(-pr_score, -conf[idx], precision[:,0])
    for j in range(tp.shape[1]):
      ap[i,j] = compute_ap(recall[:,j], precision[:,j])
  f1 = 2 * p * r / (p + r + 1e-5)
  return p, r, ap, f1, ucls.astype(np.int32)

def compute_ap(recall, precision, mode='interp'):
  """
  Compute the average precision (AP) by the area under the curve (AUC) \
  of the Recall x Precision curve.

  Args:
    recall: Recall of the predicted bounding boxes. [shape=(N,)]
    precision: Precision of the predicted bounding boxes. [shape=(N,)]
    mode: The mode of calculating the area. ['continue' or 'interp']
      interp: 101-point interpolation (COCO: https://cocodataset.org/#detection-eval).
      continue: all the point where `recall` changes.
  
  Return:
    ap: The area under the `recall` x `precision` curve.
  """
  # Add sentinel values to begin and end
  r = np.concatenate([(0.0,), recall, (min(recall[-1]+1e-5, 1.0),)])
  p = np.concatenate([(0.0,), precision, (0.0,)])
  # Compute the precision envelope
  p = np.flip(np.maximum.accumulate(np.flip(p)))

  if mode == 'interp':
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, r, p), x)
  elif mode == 'continue':
    i = np.where(r[1:]!=r[:-1])[0]
    # ap = np.sum((r[i+1] - r[i]) * p[i+1])  # p[i] == p[i+1]
    ap = np.sum((r[i+1] - r[i]) * p[i])  # p[i] == p[i+1]

  # Image DEBUG
  # plt.subplot(121)
  # plt.plot(recall, precision)
  # plt.subplot(122)
  # plt.plot(r, p)
  # plt.title(f"AP={ap:.5f}")
  # plt.show()
  return ap
