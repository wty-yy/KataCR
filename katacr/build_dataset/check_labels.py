# -*- coding: utf-8 -*-
'''
@File    : check_labels.py
@Time    : 2023/11/11 09:55:40
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
Open [`labelme`](https://github.com/wkentaro/labelme) with this suffix
(check for each param with `--help`): 
  `labelme --nodata --autosave --keep-prev`

The label file `imagename.json` struct should be:
{
  ...,
  'shapes': [{'label': str, 'points': [(x1,y1), (x2,y2)], 'shape_type': 'rectangle'}, ...]
  'imageHeight': int,
  'iamgeWidth': int,
  ...,
}

Let's check whether the 'label' for each file is satisfied with the
`Images annotate logs.md` in CR Dataset
'''
import json
from katacr.utils.related_pkgs.utility import *
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.constants.label_list import unit_list, unit2idx
from katacr.constants.state_list import state2idx

def build_label_txt(path: Path):  # with VOC bbox parameters and bbox states
    with open(path, 'r') as file:
      d = json.load(file)
    h, w = d['imageHeight'], d['imageWidth']

    name = path.name.split('.')[0]
    path_label = path.parent.joinpath(f"{name}.txt")
    file = open(path_label, 'w')
    for bbox in d['shapes']:
      cls, *states = bbox['label'].split('_')
      label = [0 for _ in range(12)]  # cls, x, y, w, h, state0, ..., state6
      # state
      if cls[-1] in ['0', '1']:  # belong
        cls, states = cls[:-1], [cls[-1]] + states
      if cls not in unit_list:
        raise Exception(f"'{path}' label '{cls}' is not available!")
      label[0] = unit2idx[cls]
      for state in states:
        state_cls, id = state2idx[state]
        label[state_cls+5] = id
      # x, y, w, h
      (x1, y1), (x2, y2) = bbox['points']
      bx, by = (x1+x2)/2/w, (y1+y2)/2/h
      bw, bh = abs(x1-x2)/w, abs(y1-y2)/h
      label[1:5] = bx, by, bw, bh

      file.write(" ".join([f"{x}" if type(x) == int else f"{x:.4f}" for x in label]) + "\n")
    file.close()
    return len(d['shapes'])

if __name__ == '__main__':
  path_manager = PathManager()
  paths = path_manager.sample(subset='images', part=2, regex=r'^\d+.json')
  max_path, max_bbox_num = None, 0
  for path in paths:
    num = build_label_txt(path)
    if num > max_bbox_num:
      max_bbox_num = num
      max_path = path
  print("Maximum bbox number:", max_bbox_num)
  print("with path:", max_path)
