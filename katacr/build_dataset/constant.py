# -*- coding: utf-8 -*-
'''
@File    : constant.py
@Time    : 2023/11/09 10:39:22
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
This file is used to define the clip rate for each part.
'''
from pathlib import Path

image_size = (592, 1280)
path_logs = Path(__file__).parents[2].joinpath("logs")
path_logs.mkdir(exist_ok=True)
path_features = Path(__file__) / "katacr/features"
# path_videos = Path("/home/yy/Coding/datasets/CR/fast_pig_2.6")
# path_dataset = Path("/home/wty/Coding/datasets/CR")
path_dataset = Path("/home/yy/Coding/datasets/CR")
assert path_dataset.exists(), "Dataset not exist!"

image_size_part2 = (568, 896)
split_bbox_params = {
  'part1': (0.835, 0.074, 0.165, 0.025),  # just time
  # 'part1': {  # number ocr
  #   'time': (0.835, 0.074, 0.165, 0.025),
  #   'hp0':  (0.166, 0.180, 0.090, 0.020),
  #   'hp1':  (0.755, 0.183, 0.090, 0.020),
  #   'hp2':  (0.515, 0.073, 0.090, 0.020),
  #   'hp3':  (0.162, 0.617, 0.090, 0.020),
  #   'hp4':  (0.756, 0.617, 0.090, 0.020),
  #   'hp5':  (0.511, 0.753, 0.090, 0.020),
  # },
  'part2': (0.021, 0.073, 0.960, 0.700),  # battle field, image size: (568, 896)
  'part3': (0.000, 0.821, 1.000, 0.179),  # card table
  'part4': {  # center word ocr
    'up': (0.100, 0.340, 0.800, 0.070),
    'mid': (0.180, 0.410, 0.650, 0.050),
  },
  'part2_watch_2400p': (0.024, 0.205, 0.954, 0.676),  # 1080x2400
  'part2_2400p': (0.020, 0.085, 0.960, 0.684),
}

mse_feature_match_threshold = 0.03
text_features_episode_end = ['match', 'over', 'break']
text_confidence_threshold = 0.005
MAX_NUM_BBOXES = 100  # 36 in OYASSU_20210528, 42 in OYASSU_20230305
