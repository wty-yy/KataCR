from katacr.utils.related_pkgs.utility import *

path_dataset = Path("/home/wty/Coding/datasets/CR")

import moviepy.editor as mp
from PIL import Image
from katacr.build_train_dataset.split_parts import process_part2
from tqdm import tqdm
def extract_part2(
    path_video: Path, path_save: Path,
    split_time: float = 0.5
):
  clip = mp.VideoFileClip(str(path_video))
  fps, duration = clip.fps, clip.duration
  print("process:", path_video)
  for i in tqdm(range(int(duration / split_time))):
    t = i * split_time
    image = Image.fromarray(process_part2(clip.get_frame(t)))
    image.save(str(path_save.joinpath(f"{int(t*fps)}.jpg")))

if __name__ == '__main__':
  from datapath_manager import PathManager
  path_manager = PathManager(path_dataset)
  paths = path_manager.sample('video', regex="^\d+.mp4$")
  for path in paths:
    parts = list(path.parts)
    parts[6] = 'images'
    parts = parts[:7] + parts[8:-1]
    parts.append(path.name[:-4])
    path_save = Path(*parts)
    extract_part2(path, path_save)
    exit()
    
