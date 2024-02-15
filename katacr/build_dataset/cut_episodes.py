# -*- coding: utf-8 -*-
'''
@File  : split_episodes.py
@Time  : 2023/10/15 20:46:39
@Author  : wty-yy
@Version : 1.0
@Blog  : https://wty-yy.space/
@Desc  : 提取视频中的所有回合，基于底部卡牌栏判断回合的开始，OCR识别文字来判断回合的结束，从而实现对视频文件进行划分：
|  Start episode  |  End episode  |
|   card table    |  center word  |
'''
import os, sys
sys.path.append(os.getcwd())
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.build_dataset.utils.split_part import split_part
from katacr.utils import load_image_array, colorstr
import constant as const
import cv2
from katacr.ocr_text.ocr_predict import OCRText
ocr_text = OCRText()

def get_features(path_features: Path) -> Sequence[np.ndarray]:
  features = []
  for path in path_features.iterdir():
    file_name = path.name[:-4]
    if path.is_file() and path.name[-3:] == 'jpg':
      feature = {
        'feature': None,
        'x_loc_rate': None,
        'y_loc_rate': None
      }
      feature['feature'] = load_image_array(path, to_gray=True, keep_dim=False)
      for string in file_name.split('_'):
        if '=' in string:
          axis, rate = string.split('=')
          if axis != 'x' and axis != 'y':
            raise Exception(f"Don't know `{axis}` meaning in `{str(path)}`")
          feature[f"{axis}_loc_rate"] = float(rate)
      features.append(feature)
  return features

def second2time(second):
  s = int(second)
  m = int(second // 60)
  h = int(m // 60)
  ret = ""
  if h:
    ret += f"{h:02}:"
    m = int(m % 60)
  s = int(s % 60)
  ret += f"{m:02}:{s:02}"
  return ret

def split_episodes(path_video: Path):
  # clip = mp.VideoFileClip(str(path_video))
  # fps, duration = clip.fps, clip.duration
  cap = cv2.VideoCapture(str(path_video))
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  duration = frames / fps
  print(fps, frames, duration)

  print(colorstr("[Video info]") + f" fps={fps:.2f}, duration={second2time(duration)}")
  file_name = path_video.name[:-4]
  path_episodes = path_video.parent.joinpath(file_name+"_episodes")
  # if path_episodes.exists():
  #   print(f"The episodes path '{str(path_episodes)} is exists, still continue? [Enter]'"); input()
  path_episodes.mkdir(exist_ok=True)

  start_features = get_features(const.path_features.joinpath("start_episode"))

  episode_num, start_frame = 0, -1
  # bar = tqdm(clip.iter_frames(), total=int(fps*duration)+1)
  bar = tqdm(range(1, int(frames)+1))
  for frame in bar:
    flag, image = cap.read()
    if not flag: break
    hw_ratio = image.shape[0] / image.shape[1]
    if 2.22 <= hw_ratio <= 2.23:
      image = cv2.resize(image, const.image_size_222)
    elif 2.16 <= hw_ratio <= 2.17:
      image = cv2.resize(image, const.image_size)
    else:
      raise f"Error: Don't know height/weight ratio: {hw_ratio:.2f}!"
    # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if check_feature_exists(image_gray, start_features):
      bar.set_description(f"Start episode at {frame} ({second2time(frame/fps)})")
    if start_frame == -1 and check_feature_exists(image_gray, start_features):
      episode_num += 1
      start_frame = frame
    if start_frame != -1 and check_text_exists(
      list(split_part(image_gray, 4).values()),  # images
      const.text_features_episode_end  # texts
    ):
      path = path_episodes.joinpath(f"{episode_num}.mp4")
      ffmpeg_extract_subclip(str(path_video), start_frame/fps, frame/fps+1, str(path))
      print(f"Split episode{episode_num} in {second2time(start_frame/fps)}~{second2time(frame/fps)}")
      start_frame = -1
    bar.set_description(f"Process {episode_num} episode")
  # clip.close()
  cap.release()

def match_feature(image, feature: dict):
  result = cv2.matchTemplate(image, feature['feature'], cv2.TM_SQDIFF_NORMED)
  return result.min() < const.mse_feature_match_threshold

def check_feature_exists(
    image: np.ndarray,
    features: Sequence[dict]
  ) -> bool:
  for feature in features:
    if match_feature(image, feature):
      return True
  return False

import cv2
def check_text_exists(
    images: np.ndarray,
    texts: Sequence[str]
  ):
  pred_list, conf = ocr_text.predict(images)
  pred = "".join(pred_list).lower()
  if np.max(conf) < const.text_confidence_threshold:
    return False  # no text
  for text in texts:
    if text in pred:
      return True
  return False

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--path-video", type=cvt2Path, default="/home/wty/Videos/CR/WTY_20240213_2_30fps.mp4")
  args = parser.parse_args()
  split_episodes(args.path_video)
  