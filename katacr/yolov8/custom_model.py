from ultralytics.nn.tasks import DetectionModel, v8DetectionLoss
from ultralytics.utils.loss import torch, make_anchors, xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner

class CRDetectionModel(DetectionModel):
  def init_criterion(self):
    return CRDetectionLoss(self)

class CRDetectionLoss(v8DetectionLoss):

  def __init__(self, model):
    super().__init__(model)
    self.assigner = CRTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

  def preprocess(self, targets, batch_size, scale_tensor):
    # targets[...,-1] = (idx, cls, bel, xywh)
    """Preprocesses the target counts and matches with the input batch size to output a tensor."""
    if targets.shape[0] == 0:
      out = torch.zeros(batch_size, 0, 5, device=self.device)
    else:
      i = targets[:, 0]  # image index
      _, counts = i.unique(return_counts=True)
      counts = counts.to(dtype=torch.int32)
      out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
      for j in range(batch_size):
        matches = i == j
        n = matches.sum()
        if n:
          out[j, :n] = targets[matches, 1:]  # cls, bel, xywh
      out[..., 2:6] = xywh2xyxy(out[..., 2:6].mul_(scale_tensor))  # TODO: 2 columns
    return out

  def __call__(self, preds, batch):
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
      (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    # Targets (B, 1+2+4)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 2), batch["bboxes"]), 1)  # TODO: 2 columns
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((2, 4), 2)  # cls, bel, xyxy TODO
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
      pred_scores.detach().sigmoid(),
      (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
      anchor_points * stride_tensor,
      gt_labels,
      gt_bboxes,
      mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss
    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # Bbox loss
    if fg_mask.sum():
      target_bboxes /= stride_tensor
      loss[0], loss[2] = self.bbox_loss(
        pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
      )

    loss[0] *= self.hyp.box  # box gain
    loss[1] *= self.hyp.cls  # cls gain
    loss[2] *= self.hyp.dfl  # dfl gain

    return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class CRTaskAlignedAssigner(TaskAlignedAssigner):

  def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
    """
    Compute target labels, target bounding boxes, and target scores for the positive anchor points.

    Args:
      gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 2), where b is the
                batch size and max_num_obj is the maximum number of objects.
      gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
      target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                  anchor points, with shape (b, h*w), where h*w is the total
                  number of anchor points.
      fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                (foreground) anchor points.

    Returns:
      (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
        - target_labels (Tensor): Shape (b, h*w, 2), containing the target labels for
                      positive anchor points. (Return is not important)
        - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                      for positive anchor points.
        - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                      for positive anchor points, where num_classes is the number
                      of object classes.
    """

    # Assigned target labels, (b, 1)
    batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
    target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
    # target_labels = gt_labels.long().flatten()[target_gt_idx]
    target_labels = gt_labels.long().view(-1, gt_labels.shape[-1])[target_gt_idx]  # (b, h*w, 2) TODO

    # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
    target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

    # Assigned target scores
    target_labels.clamp_(0)

    # 10x faster than F.one_hot()
    target_scores = torch.zeros(
      (target_labels.shape[0], target_labels.shape[1], self.num_classes),
      dtype=torch.int64,
      device=target_labels.device,
    )  # (b, h*w, class_number)
    # target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
    # target_scores[...,-1] = (bel, class1, class2, ...)
    target_scores.scatter_(2, target_labels[...,0:1], 1)  # cls TODO
    target_scores[...,-1] = target_labels[...,1]  # bel at last idx

    fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, class_number)
    target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

    return target_labels, target_bboxes, target_scores

  def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    """Compute alignment metric given predicted and ground truth bounding boxes."""
    na = pd_bboxes.shape[-2]
    mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
    overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
    bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

    ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
    ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
    # ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
    ind[1] = gt_labels[...,0]  # b, max_num_obj TODO
    # Get the scores of each grid for each gt cls
    bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

    # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
    pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
    gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
    overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps