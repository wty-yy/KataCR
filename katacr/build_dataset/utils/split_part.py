# -*- coding: utf-8 -*-
'''
@File    : split_parts.py
@Time    : 2023/11/11 09:47:12
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : 
This script is used to split different part from the origin image.
'''
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import katacr.build_dataset.constant as const
from katacr.build_dataset.utils.datapath_manager import PathManager

def ratio2name(img):
  if isinstance(img, Image.Image): img = np.array(img)
  r = img.shape[0] / img.shape[1]
  for name, ratio in const.ratio.items():
    if ratio[0] <= r <= ratio[1]:
      return name

def extract_bbox(image, x, y, w, h, target_size=None):
  """
  - `(x, y)`: The left top proportion point of the whole image.
  - `(w, h)`: The width and height of the proportion the whole image.
  """
  shape = image.shape
  if len(shape) == 2:
    image = image[...,None]
  x, y = int(shape[1] * x), int(shape[0] * y)
  w, h = int(shape[1] * w), int(shape[0] * h)
  image = image[y:y+h, x:x+w, :]
  if len(shape) == 2: image = image[..., 0]
  if target_size is not None:
    image = cv2.resize(image, target_size, cv2.INTER_CUBIC)
  return image

def to_gray(image):
  return np.array(Image.fromarray(image).convert('L'))

def process_part(img, part: int | str, playback: bool = False, resize=True, verbose=False):
  if not isinstance(part, str):
    part = f"part{part}"
  target_size = None
  if resize:
    target_size = const.part_sizes[part]
  if playback: part += '_playback'
  name = ratio2name(img)
  part += '_' + name
  bbox_params = const.split_bbox_params[part]
  if type(bbox_params) == dict:
    ret = {}
    for key, value in bbox_params.items():
      ret[key] = extract_bbox(img, *value, target_size)
  else:
    ret = extract_bbox(img, *bbox_params, target_size)
  if not verbose: return ret
  else: return ret, bbox_params

def process_part3(img):
  from katacr.build_dataset.constant import part3_bbox_params
  params = part3_bbox_params
  ret = []
  for param in params:
    x = extract_bbox(img, *param)  # xywh for next image position
    ret.append(x)
  return ret

def preprocess_background():
  path_manager = PathManager()
  paths = sorted(path_manager.search('images', name="background", regex=r"\d+.jpg"))
  path_save = path_manager.path / "images/part2/background"
  path_save.mkdir(exist_ok=True)
  for i, path in enumerate(paths):
    img = np.array(Image.open(str(path)))
    if 1 <= i+1 <= 25:
      img = process_part(img, '2_playback_2.22')
    elif i+1 == 26:
      img = process_part(img, '2_2.22')
    Image.fromarray(img).save(str(path_save / path.name))

# def split_part2(x):  # based ratio
#   r = np.max(x.shape[:2]) / np.min(x.shape[:2])
#   for name, ratio in const.ratio.items():
#     if ratio[0] <= r <= ratio[1]:
#       if name == 'oyassu':
#         x = process_part(x, 2)
#       if name == '2400p':
#         x = process_part(x, '2_2400p')
#       break
#   return x

def split_part(x):
  split_part(x, 2)

def split_part(x, part: str | int):  # based ratio
  part = str(part)
  x = process_part(x, part)
  return x

def test():
  path_logs = const.path_logs
  path_image = "/home/yy/Pictures/ClashRoyale/card_classification/test1.jpg"
  # path_extract = path_logs.joinpath("extract_frames")
  # path_frame = path_extract.joinpath("OYASSU_20230201")
  # path_frame = path_extract.joinpath("OYASSU_20230211")
  # path_frame = path_extract.joinpath("OYASSU_20210528")
  # path_frame = path_extract.joinpath("11")

  # image = Image.open(str(path_logs.joinpath("start_frame.jpg")))
  # image = Image.open(str(path_logs.joinpath("show_king_tower_hp.jpg")))
  # image = Image.open(str(path_logs.joinpath("start_setting_behind_king_tower.jpg")))
  # image = Image.open(str(path_frame.joinpath("end_episode1.jpg")))
  # image = Image.open(str(const.path_dataset / "images/background/background26.jpg"))
  # image = Image.open(str(path_frame / "start_episode1.jpg"))
  # image = Image.open(str(path_frame / "test1.jpg"))
  # image = Image.open("/home/yy/Pictures/ClashRoyale/demos/592x1280/test1.png")
  # image = Image.open("/home/yy/Pictures/ClashRoyale/demos/576x1280/test2.png")
  # image = Image.open("/home/yy/Pictures/ClashRoyale/demos/600x1280/test1.png")
  image = Image.open(path_image)
  # import matplotlib.pyplot as plt
  # plt.imshow(image)
  # plt.show()
  image = np.array(image)
  print("Image shape:", image.shape)

  path_image_save = path_logs.joinpath("split_image")
  path_image_save.mkdir(exist_ok=True)
  # part1 = process_part(image, 1)
  # Image.fromarray(part1).save(str(path_image_save.joinpath("part1.jpg")))
  # for key, value in part1.items():
  #   Image.fromarray(value).save(str(path_image_save.joinpath(f"part1_{key}.jpg")))
  # part2 = process_part(image, 2)
  # Image.fromarray(part2).save(str(path_image_save.joinpath("part2.jpg")))
  # part3 = process_part(image, 3)
  # Image.fromarray(part3).save(str(path_image_save.joinpath("part3.jpg")))
  # part4 = process_part(image, 4)
  # for key, value in part4.items():
  #   Image.fromarray(value).save(str(path_image_save.joinpath(f"part4_{key}.jpg")))

  # part2_playback = process_part(image, '2_playback')
  # Image.fromarray(part2_watch).save(str(path_image_save / "part2_watch.jpg"))
  for i in range(2,3):
    part = Image.fromarray(process_part(image, i+1, resize=True))
    part.save(str(path_image_save / f"part{i+1}.jpg"))
    part.show()
  for i, card in enumerate(process_part3(process_part(image, 3))):
    Image.fromarray(card).save(str(Path(path_image).with_stem(Path(path_image).stem+f'_{i}')))
  # part3_2400p = process_part(image, '3_2400p')
  # Image.fromarray(part3_2400p).save(str(path_image_save / "part3_2400p.jpg"))
  # part4_2400p = process_part(image, '4_2400p')
  # for key, value in part4_2400p.items():
  #   Image.fromarray(value).save(str(path_image_save.joinpath(f"part4_{key}.jpg")))
  #   Image.fromarray(value).show()

  # import matplotlib.pyplot as plt
  # plt.figure(figsize=(5,20))
  # plt.imshow(part2_watch)
  # plt.show()

if __name__ == '__main__':
  test()
  # preprocess_background()
