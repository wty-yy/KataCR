import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.related_pkgs.utility import *
from katacr.detection.predict import Predictor
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.constants.label_list import idx2unit
import json, argparse
from PIL import Image

LABELME_VERSION = "5.4.1"  # write into LABELME json file

def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", default="YOLOv5_v0.4.5.5",
    help="The model weights in `KataCR/logs/'model_name'-checkpoints/`")
  parser.add_argument("--load-id", default="200",
    help="The id of the model weights")
  parser.add_argument("--video-name", default=None,
    help="The name of the preprocessing video in `CR/images/part2/`")
  parser.add_argument("--episode", default=None,
    help="The episode of the video in `CR/images/part2/{video_name}/`")
  args = parser.parse_args(input_args)
  return args

class AnnotationHelper:
  def __init__(self, args):
    self.args = args
    self.path_manager = PathManager()

    from katacr.detection.parser import get_args_and_writer
    self.model_args = get_args_and_writer(no_writer=True, input_args=f"--model-name {self.args.model_name} --load-id {self.args.load_id} --batch-size 1".split())
    self._load_weights()
  
  def _load_weights(self):
    from katacr.detection.model import get_state
    state = get_state(self.model_args)
    from katacr.utils.model_weights import load_weights
    state = load_weights(state, self.model_args)
    self.predictor = Predictor(self.model_args, state)

  @partial(jax.jit, static_argnums=0)
  def _predict_box(self, x):  # return (pixel xywh, conf, side, cls)
    shape = self.model_args.image_shape
    w = jnp.array([x.shape[1] / shape[1], x.shape[0] / shape[0]])
    w = jnp.r_[w, w, [1] * 3].reshape(1,1,7)
    x = jnp.array(x, dtype=jnp.float32)[None, ...] / 255.
    x = jax.image.resize(x, (1,*shape), method='bilinear')
    pbox, pnum = self.predictor.pred_and_nms(self.predictor.state, x, iou_threshold=0.4, conf_threshold=0.4, nms_multi=10)
    pbox = pbox * w
    return pbox[0], pnum[0]

  def predict(self, x: jax.Array):
    pbox, pnum = jax.device_get(self._predict_box(x))
    pbox = pbox[:pnum]
    return pbox
  
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
      pbox = self.predict(img)
      # DEBUG
      # pbox = np.array([[50, 50, 20, 20, 0.9, 0, 1], [150, 150, 50, 50, 0.8, 1, 0]])
      xyxy = np.stack([
        pbox[:, 0] - pbox[:, 2] / 2,
        pbox[:, 1] - pbox[:, 3] / 2,
        pbox[:, 0] + pbox[:, 2] / 2,
        pbox[:, 1] + pbox[:, 3] / 2,
      ], axis=-1)
      # xyxy, side, cls
      pbox = np.concatenate([xyxy, pbox[:,4:7]], axis=-1)
      shapes = []
      for i in range(pbox.shape[0]):
        x = list(pbox[i])
        if (x[2] < 280 and x[3] < 80) or (x[0] > 390 and x[3] < 120): continue
        shapes.append({
          'label': idx2unit[int(x[6])] + str(int(x[5])),
          'points': [[float(x[0]), float(x[1])], [float(x[2]), float(x[3])]],
          'group_id': None,
          'description': f"confidence: {float(x[4])}",
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
      path_json = p.parent / (p.name.rsplit('.',1)[0] + '.json')
      with path_json.open('w') as file:
        json.dump(d, file, indent=2)

if __name__ == '__main__':
  # args = parse_args("--video-name OYASSU_666_episodes".split())
  args = parse_args()
  helper = AnnotationHelper(args)
  helper.process()
