"""
config start wandb at Path.home() / ".config" / sub_dir
ultralytics.__version__ == 8.1.24
"""
from ultralytics.cfg import get_cfg
from ultralytics.engine.model import Model
import ultralytics.models.yolo as yolo
from pathlib import Path
# from ultralytics.nn.tasks import DetectionModel
from custom_model import CRDetectionModel
from custom_validator import CRDetectionValidator
from custom_trainer import CRTrainer

class YOLO_CR(Model):
  """YOLO (You Only Look Once) object detection model. (Clash Royale)"""

  def __init__(self, model="yolov8n.pt", task=None, verbose=False):
    super().__init__(model=model, task=task, verbose=verbose)

  @property
  def task_map(self):
    """Map head to model, trainer, validator, and predictor classes."""
    return {
      "detect": {
        "model": CRDetectionModel,
        "trainer": CRTrainer,
        "validator": CRDetectionValidator,
        "predictor": yolo.detect.DetectionPredictor,
      },
    }

model = YOLO_CR("yolov8n.yaml", task='detect')
model.train(**dict(get_cfg('./katacr/yolov8/ClashRoyale.yaml')))
