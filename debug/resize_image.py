from katacr.utils.related_pkgs.utility import *
from PIL import Image

img_size = (568, 896)

def resize(path: str | Path, img_size: tuple = img_size):
  if not isinstance(path, Path): path = Path(path)
  tmp = path / "tmp"
  if path.is_file():
    tmp = path.parent / "tmp"
  tmp.mkdir(exist_ok=True)
  cnt = 0
  def save_fn(p):
    Image.open(str(p)).resize(img_size).save(tmp/p.name)
    return cnt + 1
  if path.is_dir():
    for p in path.glob("*.jpg"):
      cnt = save_fn(p)
  else:
    cnt = save_fn(path)
  print(f"Total change {cnt} images.")
  return cnt

if __name__ == '__main__':
  resize("/home/wty/Coding/GitHub/KataCR/logs/segment_unit/background")