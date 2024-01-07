# -*- coding: utf-8 -*-
'''
@File    : split_parts.py
@Time    : 2023/11/11 09:47:12
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
This script is used to split different part from the origin image.
'''
from PIL import Image
import numpy as np
from pathlib import Path
import katacr.build_dataset.constant as const
from katacr.build_dataset.utils.datapath_manager import PathManager

def extract_bbox(image, x, y, w, h):
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
  return image

def to_gray(image):
  return np.array(Image.fromarray(image).convert('L'))

def process_part(image, part_id: int | str):
  part = f"part{part_id}"
  bbox_params = const.split_bbox_params[part]
  if type(bbox_params) == dict:
    ret = {}
    for key, value in bbox_params.items():
      ret[key] = extract_bbox(image, *value)
  else:
    ret = extract_bbox(image, *bbox_params)
  return ret

def preprocess_background():
  path_manager = PathManager()
  paths = path_manager.search('images', name="background", regex=r"\d+.jpg")
  path_save = path_manager.path / "images/part2/background"
  path_save.mkdir(exist_ok=True)
  for path in paths:
    img = np.array(Image.open(str(path)))
    img = process_part(img, '2_watch')
    Image.fromarray(img).save(str(path_save / path.name))

def test():
  path_logs = const.path_logs
  path_extract = path_logs.joinpath("extract_frames")
  # path_frame = path_extract.joinpath("OYASSU_20230201")
  path_frame = path_extract.joinpath("OYASSU_20230211")
  # path_frame = path_extract.joinpath("OYASSU_20210528")

  # image = Image.open(str(path_logs.joinpath("start_frame.jpg")))
  # image = Image.open(str(path_logs.joinpath("show_king_tower_hp.jpg")))
  # image = Image.open(str(path_logs.joinpath("start_setting_behind_king_tower.jpg")))
  # image = Image.open(str(path_frame.joinpath("end_episode1.jpg")))
  image = Image.open(str(const.path_dataset / "images/background/background01.jpg"))
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
  part2_watch = process_part(image, '2_watch')
  Image.fromarray(part2_watch).save(str(path_image_save / "part2_watch.jpg"))

  # import matplotlib.pyplot as plt
  # plt.figure(figsize=(5,20))
  # plt.imshow(part2_watch)
  # plt.show()

if __name__ == '__main__':
  # test()
  preprocess_background()
