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
from katacr.build_dataset.constant import path_dataset
import random

class LabelBuilder:

  def __init__(self, path_dataset=path_dataset, seed=42, val_ratio=0.2):
    self.path_dataset = path_dataset
    self.path_manager = PathManager(path_dataset)
    random.seed(seed)
    self.val_ratio = val_ratio
    self.path_const_dataset = Path(__file__).parents[1] / 'constants/dataset.py'
    self.path_part2 = self.path_dataset / "images/part2"

  @staticmethod
  def build_label_txt(path: Path, box_relative=True):
      """
        Build the `.txt` file with same folder of the `.json` file `path`.

        Args:
          Path: The `.json` file was labeled by 'labelme'.
          box_relative: If taggled, the box parameters will be relative to image. \
            [0<=elems(x,y,w,h)<=1]
        Return:
          The number of the boxes in current `path` file.
      """
      with path.open('r') as file:
        d = json.load(file)
      h, w = d['imageHeight'], d['imageWidth']

      name = path.name.split('.')[0]
      path_label = path.parent / f"{name}.txt"
      file = path_label.open('w')
      for bbox in d['shapes']:
        cls, *states = bbox['label'].split('_')
        label = [0 for _ in range(12)]  # cls, x, y, w, h, state0, ..., state6
        # state
        if cls[-1] in ['0', '1']:  # belong
          cls, states = cls[:-1], [cls[-1]] + states
        if cls not in unit_list:
          raise Exception(f"Error: '{path}' label '{cls}' is not available!")
        label[0] = unit2idx[cls]
        for state in states:
          state_cls, id = state2idx[state]
          if state not in state2idx:
            raise Exception(f"Error: '{path}' state '{state}' of '{cls}' is not available!")
          label[state_cls+5] = id
        # x, y, w, h
        (x1, y1), (x2, y2) = bbox['points']
        bx, by = round((x1+x2)/2), round((y1+y2)/2)
        bw, bh = abs(x1-x2), abs(y1-y2)
        bw = min(bw, min(bx, w - bx) * 2)
        bh = min(bh, min(by, h - by) * 2)
        if box_relative:
          bx, bw = bx / w, bw / w
          by, bh = by / h, bh / h
        label[1:5] = bx, by, bw, bh

        if not box_relative:
          file.write(" ".join([f"{x}" if type(x) == int else f"{x:.0f}" for x in label]) + "\n")
        else:
          file.write(" ".join([f"{x}" if type(x) == int else f"{x:.6f}" for x in label]) + "\n")
      file.close()
      return len(d['shapes'])

  def build_annotation(self, paths: List[Path], subset=None):
    train_size = int(len(paths) * (1 - self.val_ratio))
    p2 = self.path_part2
    if subset is not None:
      path_annotation = p2 / f"{subset}_annotation.txt"
    else:
      path_annotation = p2 / f"annotation.txt"
    file = path_annotation.open('w')
    n = len(paths)
    if subset == 'train': paths = paths[:train_size]
    elif subset == 'val': paths = paths[train_size:]
    for path in paths:
      path_img = str(path.relative_to(p2)).rsplit('.', 1)[0] + '.jpg'
      path_box = str(path.relative_to(p2)).rsplit('.', 1)[0] + '.txt'
      file.write(str(path_img) + ' ' + str(path_box) + '\n')
    size = train_size if subset == 'train' else (n - train_size if subset == 'val' else n)
    if subset is not None:
      self.dfile.write(f"{subset}_datasize = {size}\n")
    else:
      self.dfile.write(f"datasize = {size}\n")
    file.close()
    return size
  
  def build(self, verbose=True):
    self.dfile = self.path_const_dataset.open('w')
    print(f"Write train/val annotation files to {self.path_part2},\ndataset infomations to {self.path_const_dataset}")
    self.dfile.write(f"path_dataset = \"{str(self.path_part2.resolve())}\"\n")

    paths = self.path_manager.search(subset='images', part=2, regex=r'^\d+.json')
    max_path, max_box_num = None, 0
    for path in paths:
      if 'background' in str(path): continue  # Don't build background to dataset
      num = self.build_label_txt(path)
      if num > max_box_num:
        max_box_num = num
        max_path = path

    random.shuffle(paths)
    train_size = self.build_annotation(paths, subset='train')
    val_size = self.build_annotation(paths, subset='val')
    total_size = self.build_annotation(paths)
    if verbose:
      print("Dataset size:", len(paths))
      print("Maximum bbox number:", max_box_num)
      print("with path:", max_path)
      train_size = int(len(paths) * (1 - self.val_ratio))
      print("Train datasize:", train_size)
      print("Val datasize:", val_size)
  
  def build_background(self):
    paths = self.path_manager.search(subset='images', part=2, name='background', regex=r'background\d+.json')
    for path in paths:
      self.build_label_txt(path)
    print("background num:", len(paths))
  
  def close(self):
    self.dfile.close()

if __name__ == '__main__':
  label_builder = LabelBuilder()
  label_builder.build()
  # label_builder.close()
  # label_builder.build_background()
