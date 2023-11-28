import sys, os
sys.path.append(os.getcwd())

from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.related_pkgs.utility import *
from katacr.yolo.predict import predict, show_bbox
from katacr.yolo.metric import get_pred_bboxes
from katacr.build_dataset.utils.split_part import process_part

from PIL import Image
import numpy as np

def load_model_state():
  from katacr.yolo.parser import get_args_and_writer
  state_args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv4 --load-id 100".split())
  state_args.batch_size = 1
  state_args.path_cp = Path("/home/yy/Coding/GitHub/KataCR/logs/YOLOv4-checkpoints")

  from katacr.yolo.yolov4_model import get_yolov4_state
  state_args.steps_per_epoch = 10  # any number
  state = get_yolov4_state(state_args)

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
  def predict_and_nms(image: jax.Array):
    x = jnp.array(image)[None, ...]
    pred = predict(state, x, state_args.anchors, state_args.image_shape)
    pred_bboxes = get_pred_bboxes(pred, conf_threshold=0.2, iou_threshold=0.5)[0]
    return pred_bboxes
  predict_and_nms(jnp.zeros((896,568,3), dtype='float32'))
  print("Compile complete!")

  bar = tqdm(clip.iter_frames(), total=math.ceil(origin_fps * origin_duration))
  SPS_avg, fps_avg = 0, 0
  for idx, frame in enumerate(bar):
    x = process_part(frame, 2)
    x = (x / np.max(x, axis=(0,1), keepdims=True)).astype(np.float32)
    start_time = time.time()
    pred_bboxes = predict_and_nms(x)
    SPS_avg += (1/(time.time() - start_time) - SPS_avg) / (idx+1)
    image = show_bbox(x, pred_bboxes, show_image=False, use_overlay=True)
    processed_frames.append(np.array(image))
    fps_avg += (1/(time.time() - start_time) - fps_avg) / (idx+1)
    bar.set_description(f"SPS:{SPS_avg:.2f} fps:{fps_avg:.2f}")
    # if idx % 30 == 0:
    #     image.show()
    # if idx == 120:
    #   break

  processed_clip = mp.ImageSequenceClip(processed_frames, fps=30)
  processed_clip.write_videofile(output_video)

def parse_args():
  from katacr.utils.parser import cvt2Path
  parser = argparse.ArgumentParser()
  # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20210528_episodes/1.mp4"),
  parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20210528_episodes/2.mp4"),
    help="The path of the input video.")
  parser.add_argument("--path-output-video", type=cvt2Path, default=None,
    help="The path of the output video, default 'logs/processed_videos/fname_yolo.mp4'")
  args = parser.parse_args()
  if args.path_output_video is None:
    fname = args.path_input_video.name
    suffix = fname.split('.')[-1]
    fname = fname[:-len(suffix)-1]
    args.path_processed_videos = Path.cwd().joinpath("logs/processed_videos")
    args.path_processed_videos.mkdir(exist_ok=True)
    args.path_output_video = args.path_processed_videos.joinpath(fname + "_yolo" + '.' + suffix)
  return args

if __name__ == '__main__':
  args = parse_args()
  main(args)
