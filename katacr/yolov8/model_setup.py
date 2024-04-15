import yaml
from pathlib import Path
from katacr.yolov8.cfg import detection_range, max_detect_num, base_idxs, invalid_units, num_detector
from katacr.build_dataset.constant import path_dataset
from katacr.constants.label_list import idx2unit, unit2idx
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt

related_units = [
  {'tesla-evolution', 'tesla-evolution-shock'},
  {'skeleton-king-skill', 'skeleton-king'}  #, 'skeleton-king-bar'} must in base_idxs
]

path_root = Path(__file__).parents[2]
path_logs = path_root / 'logs'
path_logs.mkdir(exist_ok=True)
path_part2_dataset = str(path_dataset / "images/part2")

class MultiModelSetup:
  def __init__(self, auto=True, verbose=False, num_detector=num_detector):
    """
      auto divide class by segment images size,
      otherwise use `detection_range` in cfg.py
    """
    self.auto = auto
    self.verbose = verbose
    self.num_detector = num_detector
    if not self.auto:
      self.detection_range = detection_range
    else:
      self.detection_range = self.auto_build_detection_range()

  def auto_build_detection_range(self):
    """
    Divide class indexes by segment images size, auto divide into 3 size:
    small, medium, big.
    Each detection_range meets the following rules:
      1. Contains all classes with index in `0, ..., base_idxs-1`.
      2. The maximum number of classes doesn't exceed `max_detect_num`.
      3. The last class must be `padding_belong`.
      4. Don't contain `invalid_units`.
    """
    detection_unit = unit2idx.copy()
    total_unit = []
    for k in invalid_units + list(range(base_idxs)):
      if isinstance(k, int): k = idx2unit[k]
      if k in detection_unit:
        detection_unit.pop(k)
        if k not in invalid_units:
          total_unit.append(k)
    total_unit.extend(detection_unit.keys())
    unit2size, no_img_list = {}, []
    for k in detection_unit:
      sizes = []
      path = Path(path_dataset / "images/segment" / k)
      for p in path.glob('*.png'):
        img = Image.open(str(p))
        sizes.append(np.prod(img.size))
      if not sizes:
        print("No images in", path)
        no_img_list.append(k)
      else:
        unit2size[k] = np.mean(sizes)
    unit2size = dict(sorted(unit2size.items(), key=lambda item: item[1]))
    with open(path_logs / "units_size.csv", 'w') as file:
      writer = csv.writer(file)
      writer.writerow(['Unit', 'Size'])
      for u, sz in unit2size.items():
        writer.writerow([u, float(sz)])
    detection_range = {}
    rank_list = list(unit2size.keys())
    units, n = rank_list, len(unit2size.keys())
    step = min((n + 2) // self.num_detector, max_detect_num-1)
    for i in range(self.num_detector):
      tmp_list = list(range(base_idxs)) + [unit2idx[u] for u in rank_list[:step]]
      for k in invalid_units:
        ki = unit2idx[k]
        if ki in tmp_list: tmp_list.remove(ki)
      if unit2idx['skeleton-king'] not in tmp_list:
        tmp_list.remove(unit2idx['skeleton-king-bar'])
      rank_list = rank_list[step:]
      for u in tmp_list:
        for related in related_units:
          if idx2unit[u] in related:
            for x in related:
              if x in rank_list:
                rank_list.remove(x)
                tmp_list.append(unit2idx[x])
      detection_range[f'detector{i+1}'] = tmp_list
    assert not rank_list, "The rank_list is not full, need add max_detect_num"
    if self.verbose:
      fig, ax = plt.subplots()
      ax.plot(unit2size.keys(), unit2size.values())
      for i in range(1, num_detector):
        ax.plot([units[step*i], units[step*i]], [0, 70000], 'r--')
      plt.setp(ax.get_xticklabels(), rotation=90)
      plt.show()
      print(unit2size)
      print(detection_range)
      for k, v in detection_range.items():
        print(k, [idx2unit[i] for i in v])
    print("total units:", len(total_unit), "info:", {k: len(v) for k, v in detection_range.items()})
    print("drop units:", set(unit2idx.keys()) - set(total_unit))
    if no_img_list:
      print("Don't find image:", no_img_list)
    return detection_range

  def setup_config_files(self):
    for name, idxs in self.detection_range.items():
      path_config = Path(__file__).parent / name
      path_config.mkdir(exist_ok=True)
      units = [idx2unit[idx] for idx in idxs]
      detector_idx2unit = dict(enumerate(units))
      n = len(detector_idx2unit)
      if n < max_detect_num:
        detector_idx2unit.update({i: f"padding_{i-n}" for i in range(n, max_detect_num-1)})
      detector_idx2unit.update({len(detector_idx2unit): "padding_belong"})
      data = {'path': path_part2_dataset, 'train': None, 'val': 'yolo_annotation.txt', 'test': None, 'names': detector_idx2unit}
      with path_config.joinpath('data.yaml').open('w') as file:
        yaml.dump(data, file, sort_keys=False)
      print(f"Save {name} yaml files 'data.yaml' at {str(path_config)}.")
    data = {'path': path_part2_dataset, 'train': None, 'val': 'yolo_annotation.txt', 'test': None, 'names': idx2unit}
    path_combo_config = Path(__file__).parent / 'detector_combo'
    path_combo_config.mkdir(exist_ok=True)
    with path_combo_config.joinpath('data.yaml').open('w') as file:
      yaml.dump(data, file, sort_keys=False)
    print(f"Save combo yaml files 'data.yaml' at {str(path_combo_config)}.")

if __name__ == '__main__':
  setup = MultiModelSetup(auto=True, verbose=True, num_detector=num_detector)
  setup.setup_config_files()