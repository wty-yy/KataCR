# -*- coding: utf-8 -*-
'''
@File    : extract_part2.py
@Time    : 2023/11/09 10:54:37
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
This script is used to extract all frame with `split time` seconds,
it can be used after `cut_episodes.py`in CR dataset.
The path tree struct seems like:
- Videos (Split by episode):
  datasets/CR/videos/desk_name/video_name/episode_id.mp4
- Images (Extract the part from image):
  datasets/CR/images/part_id/video_name/episode_id/frame_id.jpg
'''
from katacr.utils.related_pkgs.utility import *

path_dataset = Path("/home/wty/Coding/datasets/CR")

import moviepy.editor as mp
from PIL import Image
from katacr.build_train_dataset.split_parts import process_part
from tqdm import tqdm
def extract_part(
    path_video: Path,
    path_parts: List[str],
    split_time: float = 0.5,
    part_ids: List[int] = [1,2,3]
):
  clip = mp.VideoFileClip(str(path_video))
  fps, duration = clip.fps, clip.duration
  print("process:", path_video)
  path_saves = []
  for id in part_ids:
    parts = path_parts.copy()
    parts[-3] += f"/part{id}"
    path_save = Path(*parts)
    path_save.mkdir(parents=True, exist_ok=True)
    path_saves.append(path_save)
  for i in tqdm(range(int(duration / split_time))):
    t = i * split_time
    origin_image = clip.get_frame(t)
    for id in part_ids:
      path_save_file = path_saves[id-1].joinpath(f"{int(t*fps)}.jpg")
      if path_save_file.exists():
        print(f"the file '{path_save_file}' exists, skip processing it.")
        continue
      # print(process_func[id](origin_image))
      image = Image.fromarray(process_part(origin_image, id))
      image.save(str(path_save_file))

if __name__ == '__main__':
  from datapath_manager import PathManager
  path_manager = PathManager(path_dataset)
  paths = path_manager.sample('video', regex="^\d+.mp4$")
  for path in paths:
    parts = list(path.parts)
    parts[-4] = 'images'
    parts = parts[:-3] + parts[-2:-1]
    parts.append(path.name[:-4])
    # path_save = Path(*parts)
    # print(path_save)
    # break
    extract_part(path, path_parts=parts, part_ids=[1,2,3])
    break
    
