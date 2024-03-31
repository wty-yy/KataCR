import cv2
from katacr.build_dataset.utils.split_part import process_part
from pathlib import Path
from tqdm import tqdm

paths = [
  # "/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/OYASSU_20230203_episodes/2.mp4"
  "/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/detection_test/small_units_test_30fps.mp4"
]
path_root = Path(__file__).parents[3]
path_logs = path_root / 'logs'
path_logs.mkdir(exist_ok=True)
path_split_video = path_logs / 'split_video'
path_split_video.mkdir(exist_ok=True)

def main():
  path_logs.parts[:-2]
  writer = None
  for p in paths:
    cap = cv2.VideoCapture(p)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
      flag, img = cap.read()
      if not flag: break
      img = process_part(img, 2)
      h, w = img.shape[:2]
      if writer is None:
        # (.mp4, mp4v) 13s, (.avi, MJPG) 30s
        path_save = str(path_split_video.joinpath('_'.join(Path(p).parts[-2:])).with_suffix('.mp4'))
        writer = cv2.VideoWriter(path_save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
      # cv2.waitKey(0)
      # cv2.imshow("IMG", img)
      writer.write(img)
    writer.release()
    cap.release()

if __name__ == '__main__':
  main()