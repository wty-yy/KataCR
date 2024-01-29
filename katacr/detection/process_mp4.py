import sys, os
sys.path.append(os.getcwd())

from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.related_pkgs.utility import *
from katacr.detection.predict import Predictor
from katacr.utils.detection.data import show_box
# from katacr.detection.metric import nms_boxes_and_mask
from katacr.build_dataset.utils.split_part import process_part

from PIL import Image
import numpy as np

def load_model_state():
  from katacr.detection.parser import get_args_and_writer
  state_args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv5_v0.4.4 --load-id 100 --batch-size 1".split())

  from katacr.detection.model import get_state
  state = get_state(state_args)
  state = state.replace(params={}, batch_stats={}, grads={})

  from katacr.utils.model_weights import load_weights
  state = load_weights(state, state_args)
  return state, state_args

def main(args):
  state, state_args = load_model_state()

  import moviepy.editor as mp
  input_video = str(args.path_input_video)
  output_video = str(args.path_output_video)

  clip = mp.VideoFileClip(input_video)
  image_size = clip.size
  origin_fps = clip.fps
  origin_duration = clip.duration

  processed_frames = []

  print("Compile XLA...")
  predictor = Predictor(state_args, state)
  @jax.jit
  def preprocess(x):
    w = jnp.array([x.shape[1] / state_args.image_shape[1], x.shape[0] / state_args.image_shape[0]])
    w = jnp.r_[w, w, [1] * 3].reshape(1,1,7)
    x = jnp.array(x, dtype=jnp.float32)[None, ...] / 255.
    x = jax.image.resize(x, (1,*state_args.image_shape), method='bilinear')
    pbox, pnum = predictor.pred_and_nms(state, x, iou_threshold=0.4, conf_threshold=0.5, nms_multi=10)
    pbox = pbox * w
    return pbox[0], pnum[0]
  def predict(x: jax.Array):
    pbox, pnum = jax.device_get(preprocess(x))
    pbox = pbox[:pnum]
    return pbox
  predict(np.zeros((*image_size,3), dtype=np.uint8))
  print("Compile complete!")

  bar = tqdm(clip.iter_frames(), total=math.ceil(origin_fps * origin_duration))
  SPS_avg, fps_avg = 0, 0
  for idx, frame in enumerate(bar):
    x = process_part(frame, 2)
    start_time = time.time()
    pbox = predict(x)
    SPS_avg += (1/(time.time() - start_time) - SPS_avg) / (idx+1)
    image = show_box(x, pbox, verbose=False, use_overlay=True, show_conf=True)
    processed_frames.append(np.array(image))
    fps_avg += (1/(time.time() - start_time) - fps_avg) / (idx+1)
    bar.set_description(f"SPS:{SPS_avg:.2f} fps:{fps_avg:.2f}")
    # image.show()
    # break
    # if idx % 30 == 0:
    #     image.show()
    # if idx == 120:
    #   break

  processed_clip = mp.ImageSequenceClip(processed_frames, fps=30)
  processed_clip.write_videofile(output_video)

def parse_args():
  from katacr.utils.parser import cvt2Path, str2bool
  parser = argparse.ArgumentParser()
  parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20210528_episodes/1.mp4"),
  # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20230212_episodes/4.mp4"),
  # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20230305_episodes/4.mp4"),
  # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20230224_episodes/2.mp4"),
  # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20210528_episodes/2.mp4"),
    help="The path of the input video.")
  parser.add_argument("--path-output-video", type=cvt2Path, default=None,
    help="The path of the output video, default 'logs/processed_videos/fname_detection.mp4'")
  parser.add_argument("--show-confidence", type=str2bool, default=False,
    help="Show the confidence with the label.")
  args = parser.parse_args()
  if args.path_output_video is None:
    fname = args.path_input_video.parent.name.rsplit('_',1)[0] + '_' + args.path_input_video.name
    suffix = fname.split('.')[-1]
    fname = fname[:-len(suffix)-1]
    args.path_processed_videos = Path(__file__).parents[2] / "logs/processed_videos"
    args.path_processed_videos.mkdir(exist_ok=True)
    args.path_output_video = args.path_processed_videos / (fname + "_yolov5_v0.4.4" + '.' + suffix)
  return args

if __name__ == '__main__':
  args = parse_args()
  main(args)
