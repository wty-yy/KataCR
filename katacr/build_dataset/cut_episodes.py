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
2024.4.11: 将OCR识别从CRNN转为paddleOCR，识别目标修改如下：右上角时间以及中间区域的文字，
判断episode方式如下：
  Start episdoe: Time <= 10s or 'fight' in center_text
  End episode: Had started a episode and (
    ['match', 'over', 'break'] in center_text
    or satisfy start episode conditions
'''
import os, sys
sys.path.append(os.getcwd())
from katacr.utils.ffmpeg.ffmpeg_tools import ffmpeg_extract_subclip
from katacr.utils.related_pkgs.utility import *
from katacr.build_dataset.utils.split_part import split_part
from katacr.utils import load_image_array, colorstr, second2str
import katacr.build_dataset.constant as const
import cv2
# from katacr.ocr_text.ocr_predict import OCRText
from katacr.ocr_text.paddle_ocr import OCR

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

def get_center_texts(ocr: OCR, img: Sequence[np.ndarray], show=False):
  h, w = img.shape
  center_h = int(h * 0.43)
  center_w = int(w / 2)
  target_h = int(h * 0.23)
  x0,y0,x1,y1 = [0, center_h-target_h//2, w, center_h+target_h//2]
  center_img = img[y0:y1, x0:x1]
  if show:
    cv2.imshow('center', center_img)
    cv2.waitKey(1)
  results = ocr(center_img, gray=True)[0]
  if results is None: return []
  recs = [info[1][0] for info in results]
  return recs

def get_time(ocr: OCR, img, time_split=True, show=False):
  if not time_split:
    time_img = img[:150, -150:]
  else: time_img = img
  results = ocr(time_img, gray=True)[0]
  if show:
    # print("OCR results:", results)
    cv2.imshow('time', time_img)
    cv2.waitKey(1)
  if results is None: return math.inf
  stage = m = s = None
  for info in results:
    det, rec = info
    rec = rec[0].lower()
    if 'left' in rec:
      stage = 0
    if 'over' in rec:
      stage = 1
    if (':' in rec) or ('：' in rec):
      try:
        m, s = rec.split(':' if ':' in rec else '：')
        m = int(m.strip())
        s = int(s.strip())
      except ValueError:
        m = s = None
  if stage is None or m is None or s is None: return math.inf
  t = m * 60 + s
  if stage == 0:
    return 180 - t
  return 180 + 120 - t

class Cutter:

  def __init__(self):
    self.ocr = OCR()

  def split_episodes(self, path_video: Path, show=True, interval=10):
    cap = cv2.VideoCapture(str(path_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps
    print(fps, frames, duration)

    print(colorstr("[Video info]") + f" fps={fps:.2f}, duration={second2str(duration)}")
    path_video = Path(path_video)
    file_name = path_video.name[:-4]
    path_episodes = path_video.parent.joinpath(file_name+"_episodes")
    # if path_episodes.exists():
    #   print(f"The episodes path '{str(path_episodes)} is exists, still continue? [Enter]'"); input()
    path_episodes.mkdir(exist_ok=True)

    # start_features = get_features(const.path_features.joinpath("start_episode"))

    episode_num, start_frame = 0, -1
    # bar = tqdm(clip.iter_frames(), total=int(fps*duration)+1)
    bar = tqdm(range(1, int(frames//interval)))
    for frame in bar:
      for _ in range(interval):
        falg = cap.grab()
      flag, image = cap.retrieve()
      if not flag: break
      hw_ratio = image.shape[0] / image.shape[1]
      if 2.22 <= hw_ratio <= 2.23:
        image = cv2.resize(image, const.image_size_222)
      elif 2.16 <= hw_ratio <= 2.17 or 2.13 <= hw_ratio <= 2.14:
        image = cv2.resize(image, const.image_size)
      else:
        raise f"Error: Don't know height/weight ratio: {hw_ratio:.2f}!"
      # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # if start_frame == -1 and check_feature_exists(image_gray, start_features):
      #   episode_num += 1
      #   start_frame = frame
      # if start_frame != -1 and check_text_exists(
      #   list(split_part(image_gray, 4).values()),  # images
      #   const.text_features_episode_end  # texts
      # ):
      time = get_time(self.ocr, image_gray, time_split=False, show=show)
      # print("Time now", time)
      center_texts = get_center_texts(self.ocr, image_gray, show=show)
      def check_has_texts(texts):
        for i in center_texts:
          for j in texts:
            if j.lower() in i.lower(): return True
        return False
      # print('check1', check_has_texts(['fight']))
      # print('check2', check_has_texts(const.text_features_episode_end))
      start_flag = check_has_texts(['fight']) or time < 10
      if start_frame != -1 and (
        check_has_texts(const.text_features_episode_end)
        or ((frame - start_frame) * interval / fps > 15 and start_flag)  # WE think each episode length will longer than 30s
        or (frame == int(frames//interval)-1)):
        dt_end = 0
        if start_flag:
          dt_end = -1.4
        path = path_episodes.joinpath(f"{episode_num}.mp4")
        ffmpeg_extract_subclip(str(path_video), start_frame*interval/fps, frame*interval/fps+1+dt_end, str(path))
        print(f"Split episode{episode_num} in {second2str(start_frame*interval/fps)}~{second2str(frame*interval/fps)}")
        start_frame = -1
      if start_frame == -1 and start_flag:
        episode_num += 1
        start_frame = frame
      bar.set_description(f"Process {episode_num} episode" + f", start at {second2str(start_frame*interval/fps)}" if start_frame != -1 else "")
      if show:
        info = f"time: {time},\ncenter text: {center_texts}"
        for i, line in enumerate(info.split('\n')):
          y = (i+1) * 25
          image = cv2.putText(image, line, (0, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        cv2.imshow('ocr', image)
        cv2.waitKey(1)
    cap.release()

  # from katacr.ocr_text.ocr_predict import OCRText
  # ocr = OCR()
  # def check_text_exists_jax(
  #     images: np.ndarray,
  #     texts: Sequence[str]
  #   ):
  #   pred_list, conf = ocr(images)
  #   pred = "".join(pred_list).lower()
  #   if np.max(conf) < const.text_confidence_threshold:
  #     return False  # no text
  #   for text in texts:
  #     if text in pred:
  #       return True
  #   return False

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument("--path-video", type=cvt2Path, default="/home/yy/Coding/datasets/CR/videos/WTY_20240213.mp4")
  parser.add_argument("--path-video", type=cvt2Path, default="/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240512_134920_golem_ai.mp4")
  args = parser.parse_args()
  cutter = Cutter()
  cutter.split_episodes(args.path_video, show=False)
  # for p in Path("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6").glob('*.mp4'):
  #   cutter.split_episodes(str(p), show=True)
  