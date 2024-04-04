from katacr.build_dataset.constant import path_logs, Path
from katacr.constants.card_list import card_list
from PIL import Image
import pillow_avif
import numpy as np
path_card_cls = path_logs / "card_cls"
path_save = Path("/home/yy/Coding/datasets/Clash-Royale-Dataset/images/card_classification")

def check():
  cl = card_list.copy()
  for p in path_card_cls.glob('*'):
    name = p.stem
    if '-ev1' in name:
      name = name.replace('ev1', 'evolution')
      p.rename(p.with_stem(name))
      print(f"Rename {p}")
    if name in cl:
      cl.remove(name)
    else:
      raise KeyError(f"Don't find name {name} in card_list")

def tight_and_rename():
  for p in path_card_cls.glob('*'):
    img = np.array(Image.open(p))
    back = img[...,3]
    img = img[...,:3]
    img[back < 220] = 0
    Image.fromarray(img).save(path_save/p.with_suffix('.jpg').name)

if __name__ == '__main__':
  check()
  tight_and_rename()
