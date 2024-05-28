from katacr.yolov8.custom_validator import CRDetectionValidator
from ultralytics.engine.validator import smart_inference_mode, torch, callbacks, check_imgsz, LOGGER, check_det_dataset, Profile, json, colorstr, TQDM
from ultralytics.models.yolo.detect.val import converter, os, ConfusionMatrix
from katacr.yolov8.train import YOLO_CR, get_cfg, Path
from katacr.yolov8.combo_detect import unit2idx, idx2unit, torchvision

path_root = Path(__file__).parents[2]

path_detectors = [  # The combo detectors to evaluate
  # '/home/yy/Coding/GitHub/KataCR/runs/detector1_v0.7.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector2_v0.7.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector1_v0.7.1.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector2_v0.7.1.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector3_v0.7.1.pt',
  # './runs/detect/detector1_v0.7.1/weights/best.pt',
  # './runs/detect/detector2_v0.7.1/weights/best.pt',
  # './runs/detect/detector3_v0.7.1/weights/best.pt',
  # './runs/detect/detector1_v0.7.7/weights/best.pt',
  # './runs/detect/detector2_v0.7.7/weights/best.pt',
  # './runs/detect/detector3_v0.7.7/weights/best.pt',
  path_root / './runs/detector1_v0.7.13.pt',
  path_root / './runs/detector2_v0.7.12.2.pt',
  # path_root / './runs/detector2_v0.7.12.2.pt',
  # path_root / './runs/detector1_v0.7.13_tri.pt',
  # path_root / './runs/detector2_v0.7.13_tri.pt',
  # path_root / './runs/detector3_v0.7.13_tri.pt',
]

class ComboDetectionValidator(CRDetectionValidator):

  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
    """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
    gets priority).
    """
    self.data = check_det_dataset(self.args.data)
    self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

    self.run_callbacks("on_val_start")
    dt = (
      Profile(device=self.device),
      Profile(device=self.device),
      Profile(device=self.device),
      Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
    self.init_metrics()
    self.jdict = []  # empty before each val
    for batch_i, batch in enumerate(bar):
      self.run_callbacks("on_val_batch_start")
      self.batch_i = batch_i
      # Preprocess
      with dt[0]:
        batch = self.preprocess(batch)

      # Inference
      with dt[1]:
        preds = model(batch["img"])

      self.update_metrics(preds, batch)
      if self.args.plots and batch_i < 3:
        self.plot_val_samples(batch, batch_i)
        self.plot_predictions(batch, preds, batch_i)

      self.run_callbacks("on_val_batch_end")
    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()
    self.run_callbacks("on_val_end")
    LOGGER.info(
      "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
      % tuple(self.speed.values())
    )
    if self.args.save_json and self.jdict:
      with open(str(self.save_dir / "predictions.json"), "w") as f:
        LOGGER.info(f"Saving {f.name}...")
        json.dump(self.jdict, f)  # flatten and save
      stats = self.eval_json(stats)  # update stats
    if self.args.plots or self.args.save_json:
      LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
    return stats

  def init_metrics(self):
    """Initialize evaluation metrics for YOLO."""
    val = self.data.get(self.args.split, "")  # validation path
    self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
    self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
    self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
    self.names = self.data['names']  # TODO: remove from model
    self.nc = len(self.data['names'])
    self.metrics.names = self.names
    self.metrics.plot = self.args.plots
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
    self.seen = 0
    self.jdict = []
    self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

class ComboModel:
  def __init__(self, path_detectors=path_detectors):
    print("Using detectors:", path_detectors)
    self.models = [YOLO_CR(str(p)) for p in path_detectors]
  
  def __call__(self, x):
    results = [m.predict(x, verbose=False, conf=0.001) for m in self.models]
    preds = []
    for i in range(x.shape[0]):
      pred = []
      for result in results:
        p = result[i]
        boxes = p.logits_boxes.clone()
        # print(boxes)
        for j in range(len(boxes)):
          boxes[j, 5] = unit2idx[p.names[int(boxes[j, 5])]]
          pred.append(boxes[j])
      if not pred:
        pred = torch.zeros(0, 7)
      else:
        pred = torch.cat(pred, 0).reshape(-1, 7)
      idx = torchvision.ops.nms(pred[:, :4], pred[:, 4], iou_threshold=0.7)
      pred = pred[idx].cpu()
      preds.append(pred)
    return preds

if __name__ == '__main__':
  combo_model = ComboModel()
  cfg = dict(get_cfg('./katacr/yolov8/ClashRoyale.yaml'))
  name = 'detector_combo'
  cfg['name'] = name + '_' + cfg['name']
  cfg['data'] = Path(__file__).parent / f"{name}/data.yaml"
  combo_validator = ComboDetectionValidator(args=cfg)
  combo_validator(model=combo_model)

