import katacr.build_train_dataset.constant as const
from katacr.utils.related_pkgs.utility import *

def parse_args():
  parser = argparse.ArgumentParser(description=
    "INFO: To split video episode frames to each part, "
    "the processing path is `path_videos/video_name/id.mp4`, "
    "where `id` starts from `start_id` and traverse all the remaining ids in "
    "`path_videos/video_name/`"
  )
  parser.add_argument("--path-videos", type=cvt2Path, default=const.path_videos,
    help="the path of the videos, contains subdirectories with YouTube video name to each episode")
  parser.add_argument("--video-name", type=cvt2Path, default="OYASSU_20210528",
    help="the path of the videos name (`video author + release date`)")
  parser.add_argument("--start-id", type=int, default=1,
    help="the episode id to start processing, `path_videos/videos`")
  parser.add_argument("--frame-gap", type=int, default=15,
    help="the length of gap between frames")
  args = parser.parse_args()
  args.path_episodes = args.path_videos.joinpath(f"{args.video_name}_episodes")
  return args

import moviepy.editor as mp
from PIL import Image
from split_parts import process_part2, process_part3
def split_video_to_parts(
    path_video: Path,
    path_save: Path,
    prefix_name: str,
    gap: int
):
  def process_and_save(func: Callable, path_save: Path):
    image = func(frame)
    path_save = path_save.joinpath(prefix_name+f"{idx:05}"+'.jpg')
    Image.fromarray(image).save(str(path_save))

  func_with_path = [
    (lambda x: x, path_save.joinpath("origin")),
    (process_part2, path_save.joinpath("part2")),
    (process_part3, path_save.joinpath("part3")),
  ]
  for _, path in func_with_path: path.mkdir(exist_ok=True)

  clip = mp.VideoFileClip(str(path_video))
  fps, duration = clip.fps, clip.duration
  bar = tqdm(clip.iter_frames(), total=int(fps*duration))
  for idx, frame in enumerate(bar):
    if idx % gap == 0:
      for func, path in func_with_path:
        process_and_save(func, path)

if __name__ == '__main__':
  args = parse_args()
  paths = sorted(list(args.path_episodes.iterdir()))
  path_save = const.path_logs.joinpath(f"split_image/")
  path_save.mkdir(exist_ok=True)
  for path_video in paths:
    assert(path_video.name[-4:] == '.mp4')
    episode_id = int(path_video.name[:-4])
    if episode_id < args.start_id: continue
    prefix_name = f"{args.video_name}_{episode_id}_"
    split_video_to_parts(path_video, path_save, prefix_name, args.frame_gap)
