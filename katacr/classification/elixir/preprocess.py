from pathlib import Path
from katacr.build_dataset.constant import path_dataset
import cv2
import glob

path = path_dataset / "images/elixir_classification"

def resize_images():
  for p in path.rglob('*'):
    if not p.is_file(): continue
    img = cv2.imread(str(p))
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(p.with_suffix('.jpg')), img)

if __name__ == '__main__':
  resize_images()