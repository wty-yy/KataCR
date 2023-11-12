from katacr.utils.related_pkgs.utility import *
from katacr.build_dataset.utils.datapath_manager import PathManager

def get_bbox_size():
  path_manager = PathManager()
  paths = path_manager.sample('images', part=2, regex=r".txt")
  ret = []
  for path in paths:
    with open(path, 'r') as file:
      params = file.read().split('\n')[:-1]
    for param in params:
      parts = param.split(' ')
      ret.append(np.array((parts[3], parts[4])))
    # print(path)
  return np.array(ret)

if __name__ == '__main__':
  print(get_bbox_size().shape)