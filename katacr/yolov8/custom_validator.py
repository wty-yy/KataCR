from ultralytics.models.yolo.detect.val import DetectionValidator, output_to_target, torch, Path
from ultralytics.utils import ops
from katacr.yolov8.custom_utils import plot_images, non_max_suppression

class CRDetectionValidator(DetectionValidator):
  def plot_val_samples(self, batch, ni):
    plot_images(
      batch["img"],
      batch["batch_idx"],
      batch["cls"],  # TODO: (B,2)
      batch["bboxes"],
      paths=batch["im_file"],
      fname=self.save_dir / f"val_batch{ni}_labels.jpg",
      names=self.names,
      on_plot=self.on_plot,
    )
  
  def plot_predictions(self, batch, preds, ni):
    plot_images(
      batch["img"],
      *output_to_target(preds, max_det=self.args.max_det),
      paths=batch["im_file"],
      fname=self.save_dir / f"val_batch{ni}_pred.jpg",
      names=self.names,
      on_plot=self.on_plot,
    )  # pred
  
  def postprocess(self, preds):
    """Apply Non-maximum suppression to prediction outputs."""
    return non_max_suppression(
      preds,
      self.args.conf,
      self.args.iou,
      labels=self.lb,
      multi_label=False,  # TODO: True -> False
      agnostic=self.args.single_cls,
      max_det=self.args.max_det,
    )
  
  def _prepare_batch(self, si, batch):
    """Prepares a batch of images and annotations for validation."""
    idx = batch["batch_idx"] == si
    cls = batch["cls"][idx][:,0]  # TODO: just conside cls
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"][si]
    imgsz = batch["img"].shape[2:]
    ratio_pad = batch["ratio_pad"][si]
    if len(cls):
      bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
      ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
    return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

  def update_metrics(self, preds, batch):
    """Metrics."""
    for si, pred in enumerate(preds):
      self.seen += 1
      npr = len(pred)
      stat = dict(
        conf=torch.zeros(0, device=self.device),
        pred_cls=torch.zeros(0, device=self.device),
        tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
      )
      pbatch = self._prepare_batch(si, batch)
      cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
      nl = len(cls)
      stat["target_cls"] = cls
      if npr == 0:
        if nl:
          for k in self.stats.keys():
            self.stats[k].append(stat[k])
          if self.args.plots:
            self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
        continue

      # Predictions
      if self.args.single_cls:
        pred[:, 5] = 0
      predn = self._prepare_pred(pred, pbatch)
      stat["conf"] = predn[:, 4]
      stat["pred_cls"] = predn[:, 5]

      # Evaluate
      if nl:
        stat["tp"] = self._process_batch(predn, bbox, cls)
        if self.args.plots:
          self.confusion_matrix.process_batch(predn, bbox, cls)
      for k in self.stats.keys():
        self.stats[k].append(stat[k])

      # Save
      if self.args.save_json:
        self.pred_to_json(predn, batch["im_file"][si])
      if self.args.save_txt:
        file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
        self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)