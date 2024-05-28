from ultralytics.models.yolo.detect.train import DetectionTrainer, copy, torch_distributed_zero_first, LOGGER, build_dataloader
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import colorstr, RANK
from pathlib import Path
from katacr.yolov8.custom_model import CRDetectionModel
from katacr.yolov8.custom_validator import CRDetectionValidator
from katacr.yolov8.custom_utils import plot_images
from katacr.yolov8.custom_dataset import CRDataset

class CRTrainer(DetectionTrainer):

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return a YOLO detection model."""
    model = CRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
    if weights:
      model.load(weights)
    return model
  
  def get_validator(self):
    """Returns a DetectionValidator for YOLO model validation."""
    self.loss_names = "box_loss", "cls_loss", "dfl_loss"
    return CRDetectionValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )

  def build_dataset(self, img_path, mode="train", batch=None):
    return CRDataset(
      img_path=img_path,
      imgsz=self.args.imgsz,
      cache=self.args.cache,
      # augment=mode == 'train',
      augment=False,
      hyp=self.args,
      prefix=colorstr(f"{mode} 123: "),
      # rect=self.args.rect,
      rect=True,  # TODO: set rect True, since CR Dataset is same size
      batch_size=batch,
      stride=32,
      pad=0.0,
      single_cls=False,
      classes=None,  # only include class
      fraction=1.0,
      data=self.data,
      seed=self.args.seed,  # TODO: add to generation dataset config
    )
  
  def plot_training_samples(self, batch, ni):
    plot_images(
      images=batch["img"],
      batch_idx=batch["batch_idx"],
      cls=batch["cls"],  # TODO: cls with 2 columns, (B, 2)
      bboxes=batch["bboxes"],
      paths=batch["im_file"],
      fname=self.save_dir / f"train_batch{ni}.jpg",
      on_plot=self.on_plot,
      names=self.data['names'],  # TODO: add names
    )
  
  def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
    """Construct and return dataloader."""
    assert mode in ["train", "val"]
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
      dataset = self.build_dataset(dataset_path, mode, batch_size)
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle:
      # LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
      LOGGER.info("OK ✅ 'rect=True' is compatible with CR DataLoader shuffle, setting shuffle=True")
      shuffle = True
    workers = self.args.workers if mode == "train" else self.args.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

  def plot_training_labels(self):  # TODO: Don't print generative train dataset
    pass
