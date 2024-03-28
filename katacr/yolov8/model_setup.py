import yaml
from pathlib import Path
from katacr.yolov8.cfg import detection_range, max_detect_num
from katacr.build_dataset.constant import path_dataset
from katacr.constants.label_list import idx2unit

path_part2_dataset = str(path_dataset / "images/part2")

def setup_config_files():
  for name, idxs in detection_range.items():
    path_config = Path(__file__).parent / name
    path_config.mkdir(exist_ok=True)
    units = [idx2unit[idx] for idx in idxs]
    detector_idx2unit = dict(enumerate(units))
    n = len(detector_idx2unit)
    if n < max_detect_num:
      detector_idx2unit.update({i: f"padding_{i-n}" for i in range(n, max_detect_num-1)})
    detector_idx2unit.update({len(detector_idx2unit): "padding_belong"})
    data = {'path': path_part2_dataset, 'train': None, 'val': 'annotation.txt', 'test': None, 'names': detector_idx2unit}
    with path_config.joinpath('data.yaml').open('w') as file:
      yaml.dump(data, file, sort_keys=False)
    print(f"Save {name} yaml files 'data.yaml' at {str(path_config)}.")

if __name__ == '__main__':
  setup_config_files()