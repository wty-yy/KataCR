"""
config start wandb at Path.home() / ".config" / sub_dir
ultralytics.__version__ == 8.1.24
Running with multi gpu should add ../KataCR to your PYTHONPATH in .bashrc or .zshrc
export PYTHONPATH=$PYTHONPATH:/Your/Path/KataCR
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from ultralytics.cfg import get_cfg
from ultralytics.engine.model import Model
from pathlib import Path
from katacr.yolov8.custom_model import CRDetectionModel
from katacr.yolov8.custom_validator import CRDetectionValidator
from katacr.yolov8.custom_trainer import CRTrainer
from katacr.yolov8.custom_predict import CRDetectionPredictor

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
        "predictor": CRDetectionPredictor,
      },
    }

  def track(self, source=None, stream=False, persist=False, **kwargs,) -> list:
    if not hasattr(self.predictor, "trackers"):
      from katacr.yolov8.custom_trackers import register_tracker

      register_tracker(self, persist)
    kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
    kwargs["mode"] = "track"
    return self.predict(source=source, stream=stream, **kwargs)

import argparse
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--detector", type=int, default=1, help="The training detector index.")
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  model = YOLO_CR("yolov8l.yaml", task='detect')
  cfg = dict(get_cfg('./katacr/yolov8/ClashRoyale.yaml'))
  name = f"detector{args.detector}"
  cfg['name'] = name + '_' + cfg['name']
  cfg['data'] = Path(__file__).parent / f"{name}/data.yaml"
  model.train(**cfg)
