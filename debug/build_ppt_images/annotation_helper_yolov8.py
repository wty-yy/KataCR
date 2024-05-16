# -*- coding: utf-8 -*-
'''
@File    : annotation_helper_yolov8.py
@Time    : 2024/04/12 16:41:04
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : Use combo yolov8 to help annotate labels. Save show_box image
'''
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from katacr.utils.related_pkgs.utility import *
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.constants.label_list import idx2unit
import json, argparse
from PIL import Image
from katacr.utils.detection.data import show_box

LABELME_VERSION = "5.4.1"  # write into LABELME json file

def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-version", default="v0.7.13",
    help="The model weights in `KataCR/runs/detector{i}_{model_version}`")
  parser.add_argument("--video-name", default=None,
    help="The name of the preprocessing video in `CR/images/part2/{video_name}`")
  parser.add_argument("--episode", default=None,
    help="The episode of the video in `CR/images/part2/{video_name}/{episode}`")
  args = parser.parse_args(input_args)
  return args

class AnnotationHelper:
  def __init__(self, args):
    self.args = args
    self.path_manager = PathManager()

    path_detectors = list(path_root.glob(f"runs/detector*{args.model_version}.pt"))
    from katacr.yolov8.combo_detect import ComboDetector
    self.detector = ComboDetector(path_detectors, tracker=None)

  # @partial(jax.jit, static_argnums=0)
  # def _predict_box(self, x):  # return (pixel xywh, conf, side, cls)
  #   shape = self.model_args.image_shape
  #   w = jnp.array([x.shape[1] / shape[1], x.shape[0] / shape[0]])
  #   w = jnp.r_[w, w, [1] * 3].reshape(1,1,7)
  #   x = jnp.array(x, dtype=jnp.float32)[None, ...] / 255.
  #   x = jax.image.resize(x, (1,*shape), method='bilinear')
  #   pbox, pnum = self.predictor.pred_and_nms(self.predictor.state, x, iou_threshold=0.4, conf_threshold=0.4, nms_multi=10)
  #   pbox = pbox * w
  #   return pbox[0], pnum[0]

  # def predict(self, x: jax.Array):
  #   pbox, pnum = jax.device_get(self._predict_box(x))
  #   pbox = pbox[:pnum]
  #   return pbox
  
  def _check_img_path(self, p: Path):
    path_json = p.parent / (p.name.rsplit('.',1)[0] + '.json')
    if path_json.exists():
      print(f"Found path {p.parent} image {p.name} has json file {path_json.name}, skip it!")
      return False
    return True
  
  def process(self):
    paths = self.path_manager.search('images', 2, self.args.video_name, self.args.episode, regex=r"^\d+.jpg")
    paths = [p for p in paths if self._check_img_path(p)]
    bar = tqdm(paths)
    for p in bar:
      bar.set_description(str(Path(*p.parts[-3:])))
      img = np.array(Image.open(str(p)))
      result = self.detector.infer(img, pil=True)
      # xyxy, (track_id), conf, cls, bel
      pbox = result.get_data()
      shapes = []
      for i in range(pbox.shape[0]):
        x = list(pbox[i])
        # Remove topleft and topright corner labels
        if (x[2] < 280 and x[3] < 80) or (x[0] > 390 and x[3] < 120): continue
        shapes.append({
          'label': idx2unit[int(x[-2])] + str(int(x[-1])),
          'points': [[float(x[0]), float(x[1])], [float(x[2]), float(x[3])]],
          'group_id': None,
          'description': (f"track id: {int(x[-4])}, " if len(x) == 8 else "") + f"confidence: {float(x[-3]):.4f}",
          'shape_type': 'rectangle',
          'flags': {}
        })
      d = {
        'version': LABELME_VERSION,
        'flag': {},
        'shapes': shapes,
        'imagePath': p.name,
        'imageData': None,
        'imageHeight': img.shape[0],
        'imageWidth': img.shape[1]
      }
      path_json = p.with_suffix('.json')
      with path_json.open('w') as file:
        json.dump(d, file, indent=2)
      result.show_box(save_path=(p.with_name(p.stem+'_ann.jpg')))

if __name__ == '__main__':
  args = parse_args("--video-name test --episode 5".split())
  helper = AnnotationHelper(args)
  helper.process()
