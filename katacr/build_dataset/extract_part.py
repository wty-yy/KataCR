# -*- coding: utf-8 -*-
'''
@File    : extract_part2.py
@Time    : 2023/11/09 10:54:37
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : 
This script is used to extract all frame with `split time` seconds,
it can be used after `cut_episodes.py`in CR dataset.
The path tree struct seems like:
- Videos (Split by episode):
  datasets/CR/videos/desk_name/video_name/episode_id.mp4
- Images (Extract the part from image):
  datasets/CR/images/part_id/video_name/episode_id/frame_id.jpg
2024/02/22: UPATE: extract part and resize to target image size.
2024/04/04: UPATE: add patch_detection:
- Videos:
  datasets/CR/videos/patch_detection/video_name.mp4
- Images:
  datasets/CR/images/patch_detection/video_name/frame_id.jpg
'''
from katacr.utils.related_pkgs.utility import *

# import moviepy.editor as mp
import cv2
from PIL import Image
from katacr.build_dataset.utils.split_part import process_part, process_part3
from tqdm import tqdm
from katacr.build_dataset.constant import part_sizes
def extract_part(
    path_video: Path,
    path_parts: List[str],
    # split_time: float = 0.5,
    interval: int = 15,  # 0.5 second in 30 fps
    part_mode: List[int|str] = [1,2,3],  # '1,2,3 -> part1,2,3' or 'cards'
    playback: bool = False,  # Is video playback?
    limit: tuple = (0, float('inf')),  # extract frames limit
):
  ### Check part mode ###
  for m in part_mode:
    if isinstance(m, int): assert(1 <= m <= 3)
    else:
      assert m == 'cards'
  # clip = mp.VideoFileClip()
  cap = cv2.VideoCapture(str(path_video))
  fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
  duration = frames / fps
  # fps, duration = clip.fps, clip.duration
  print("process:", path_video)
  path_saves = []
  for mode in part_mode:
    parts = path_parts.copy()
    parts[-3] += f"/part{mode}" if isinstance(mode, int) else f"/{mode}"
    path_save = Path(*parts)
    path_save.mkdir(parents=True, exist_ok=True)
    path_saves.append(path_save)
  print("Save paths:", path_saves)
  f = 0
  for i in tqdm(range(int(frames / interval))):
    cap.grab()
    flag, origin_image = cap.retrieve()
    origin_image = origin_image[...,::-1]
    h, w, _ = origin_image.shape
    for _ in range(interval-1):
      cap.grab()
    f += interval
    if not limit[0] <= f <= limit[1]: continue
    # t = i * split_time
    # origin_image = clip.get_frame(t)
    for j, mode in enumerate(part_mode):
      path_save_file = path_saves[j].joinpath(f"{int(i*interval):05}.jpg")
      if path_save_file.exists():
        print(f"the file '{path_save_file}' exists, skip processing it.")
        continue
      # print(process_func[id](origin_image))
      if isinstance(mode, int):
        image = Image.fromarray(process_part(origin_image, mode, playback=playback, resize=True))
        image.save(str(path_save_file))
      elif mode == 'cards':
        image = process_part(origin_image, 3, playback=playback, resize=True)
        cards = process_part3(image)
        for name, img in enumerate(cards):
          Image.fromarray(img).save(str(path_save_file.with_stem(path_save_file.stem+f'_{name}')))
        # image = image.resize(part_sizes['part'+str(mode)])
      # else:
      #   r = 1024 / h  # height -> 1024
      #   image = Image.fromarray(origin_image).resize((int(w*r), int(h*r)))

if __name__ == '__main__':
  from katacr.build_dataset.utils.datapath_manager import PathManager
  path_manager = PathManager()
  # paths = path_manager.sample('videos', video_name="fast_pig_2.6/OYASSU_20230212_episodes/4.mp4", regex="^\d+.mp4$")
  # paths = path_manager.search('videos', video_name="fast_pig_2.6/OYASSU_20230305_episodes/4.mp4", regex="^\d+.mp4$")
  # paths = path_manager.search('videos', video_name="fast_pig_2.6/OYASSU_20210528_episodes/5.mp4", regex="^\d+.mp4$")
  # paths = path_manager.search('videos', video_name="fast_pig_2.6/OYASSU_20230203_episodes/2.mp4", regex="^\d+.mp4$")
  # paths = path_manager.search('videos', video_name="fast_pig_2.6/WTY_20240218_episodes/1.mp4", regex="^\d+.mp4$")
  # paths = path_manager.search('videos', video_name="segment_test/WTY_20240309/WTY_20240309_barbarian.mp4")
  # paths = path_manager.search('videos', part='segment_test', video_name="WTY_20240413")
  # paths = path_manager.search('videos', part='patch_detection')
  # paths = path_manager.search('videos', video_name="/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/segment_test/WTY_20240422/1.mp4", regex="^\d+.mp4$")
  paths = path_manager.search('videos', video_name="/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/segment_test/WTY_20240506/1.mp4", regex="^\d+.mp4$")
  for path in paths:
    parts = list(path.parts)
    # datasets/CR/videos/desk_name/video_name/episode_id.mp4 -> datasets/CR/images/part_id/video_name/episode_id/frame_id.jpg
    parts[-4] = 'images'
    parts = parts[:-3] + parts[-2:-1]
    parts.append(path.stem)
    extract_part(path, path_parts=parts, part_mode=[2], interval=15, playback=False, limit=(0, float('inf')))
    
