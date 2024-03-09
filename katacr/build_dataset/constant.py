# -*- coding: utf-8 -*-
'''
@File    : constant.py
@Time    : 2023/11/09 10:39:22
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : 
This file is used to define the clip rate for each part.
'''
from pathlib import Path

image_size = (592, 1280)  # ratio: 2.16
image_size_222 = (576, 1280)  # ratio: 2.22
root = Path(__file__).parents[2]
path_logs = root / "logs"
path_logs.mkdir(exist_ok=True)
path_features = root / "katacr/features"
# path_videos = Path("/home/yy/Coding/datasets/CR/fast_pig_2.6")
# path_dataset = Path("/home/wty/Coding/datasets/CR")
path_dataset = Path("/home/yy/Coding/datasets/Clash-Royale-Dataset")
# path_dataset = Path("/data/user/wutianyang/dataset/CR")
assert path_dataset.exists(), "Dataset not exist!"

image_size_part2 = (568, 896)  # ratio: 1.57~1.58
split_bbox_params = {  # format: [x_top_left, y_top_left, width, hight]
  # Forall height/width = 2.16~2.17
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
  'part2': (0.021, 0.073, 0.960, 0.700),  # battle field, image size: (568, 896), origin: 592x1282, ratio: 2.16~2.17
  'part3': (0.000, 0.821, 1.000, 0.179),  # card table
  'part4': {  # center word ocr
    'up': (0.100, 0.340, 0.800, 0.070),
    'mid': (0.180, 0.410, 0.650, 0.050),
  },
  # Forall height/width = 2.22~2.23, 576x1280 or 450x1000
  'part2_playback_2400p': (0.024, 0.196, 0.954, 0.685),  # 1080x2400, ratio: 2.22~2.23
  'part2_2400p': (0.020, 0.090, 0.960, 0.680),
  'part3_2400p': (0.000, 0.850, 1.000, 0.150),
  'part4_2400p': {
    'up': (0.130, 0.352, 0.747, 0.051)
  },
}
ratio = {
  'part2': (1.57, 1.58),
  'oyassu': (2.16, 2.17),
  '2400p': (2.22, 2.23)
}

mse_feature_match_threshold = 0.03
text_features_episode_end = ['match', 'over', 'break']
text_confidence_threshold = 0.005
MAX_NUM_BBOXES = 200  # 36 in OYASSU_20210528, 42 in OYASSU_20230305

part_sizes = {
  'part1': (97, 32),
  'part2': (568, 896),
  'part3': (590, 229),
}
