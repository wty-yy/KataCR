"""
Used to manage the annotation and segment dataset version in part2,
it will create below files in `CR_dataset/version_info`:
(note: dataset=[annotation/segment], version=[v0.1,v0.2,...])

1. `{dataset}_{version}.csv`: The number of each label in current dataset.
2. `{dataset}_{version}_update.txt`: The change in number of each label betweens neighbor version.

用于对识别数据集(part2)进行版本管理，将在CR数据集的根目录下的`/part2/version_info`中生成如下文件：
1. annotation_{version}.csv, segment_{version}.csv：分别为标记、生成式数据集中的标签数目。
2. {dataset}_{version}_update.txt为数据集版本变换导致的标签数目变动。
"""
from pathlib import Path
from katacr.build_dataset.utils.datapath_manager import PathManager
import pandas as pd
import numpy as np
from katacr.constants.label_list import idx2unit
from katacr.utils import colorstr

class DatasetManager:
  def __init__(self):
    self.path_manager = PathManager()
    paths = sorted(list(Path(__file__).parent.glob('*.txt')))
    self.path_version = paths[-1] if len(paths) else None
    self.version = self.path_version.stem if self.path_version else None
  
  def update(self):
    paths = self.path_manager.search("images", "segment")
    self.segment = {}
    for p in paths:
      try:
        if p.stem.count('_') <= 1:  # Don't have side
          side = 0
        else: side = p.stem.split('_')[1]
        name = p.parent.name
        if name == 'old': continue  # skip old images
        if name not in self.segment:
          self.segment[name] = [0, 0]
        self.segment[name][int(side)] += 1
      except Exception as e:
        print(f"ERROR Path: {p}")
        raise e
    
    self.annotation = {}
    paths = self.path_manager.search("images", 2, regex="^\d+.txt")
    for p in paths:
      with p.open('r') as file:
        box = np.loadtxt(file)
      for b in box:
        name, side = idx2unit[int(b[0])], int(b[5])
        if name not in self.annotation:
          self.annotation[name] = [0, 0]
        self.annotation[name][side] += 1
  
  def save(self, part=['annotation', 'segment']):
    if not isinstance(part, (list, tuple)): part = [part]
    root_path = self.path_manager.path / "version_info"
    root_path.mkdir(exist_ok=True)
    for pt in part:
      data: dict = eval("self."+pt)
      print(colorstr("Process part: ") + colorstr('red', pt) + "...")
      paths = sorted(list(root_path.glob(pt + '*.csv')), key=lambda p: int(p.stem.split('_')[1].split('.')[1]))
      if len(paths) == 0:
        new_version = '0.1'
        print("Don't find any old version, start from v0.1")
      else:
        old_version = paths[-1].stem.split('_')[-1][1:]
        new_version = '0.' + str(int(old_version.split('.')[1])+1)
        print(f"Found old version {colorstr('red', 'v'+old_version)}")
        df = pd.read_csv(str(paths[-1]), index_col=0)
        same_flag, diff_count, note_file = True, 0, None
        for i in data:
          new = np.array(data.get(i, []))
          if i not in df.index:
            old = np.zeros((2,), dtype=np.int32)
          else:
            old = np.array(df.loc[i])
          if not (old == new).all():
            if same_flag:
              print("Find following difference between the two versions:")
              note_file = (root_path / f"{pt}_v{new_version}_update.txt").open('w')
              s = ' ' * 30 + f"{'Old (v'+old_version+')':>16} -> {'New (v'+new_version+')':>16} {'Count':>8}"
              note_file.write(s + '\n')
              print(s)
            diff_count += 1
            s = f"{i:<29} {str(old):>16} -> {str(new):>16} {diff_count:>8}"
            note_file.write(s + '\n')
            print(s)
            same_flag = False
        if note_file is not None: note_file.close()
        if same_flag:
          print(f"Version information is same, skip to save")
          continue

      df = pd.DataFrame(data).T
      save_path = root_path / (pt + '_v' + new_version + '.csv')
      df.to_csv(str(save_path))
      print("Save to " + colorstr('green', save_path))

if __name__ == '__main__':
  manager = DatasetManager()
  manager.update()
  manager.save()