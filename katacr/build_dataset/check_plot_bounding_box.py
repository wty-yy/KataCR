from katacr.utils.related_pkgs.utility import *
from katacr.utils.detection import plot_boxes_PIL, plot_cells_PIL
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.constants.state_list import idx2state
from katacr.constants.label_list import idx2unit
from katacr.build_dataset.constant import path_logs
from PIL import Image
import matplotlib.pyplot as plt

def get_box_colors(n):
  cmap = plt.cm.rainbow
  step = cmap.N // n
  colors = cmap([i for i in range(0, cmap.N, step)])
  colors = (colors[:, :3] * 255).astype(np.int32)
  return colors

if __name__ == '__main__':
  path_manager = PathManager()
  paths = path_manager.sample(subset='images', part=2, regex=r'^\d+.txt')
  path_save = path_logs.joinpath("label_images")
  path_save.mkdir(exist_ok=True)
  for path_txt in tqdm(paths):
    path_image = path_txt.parent.joinpath(path_txt.name[:-3]+"jpg")
    image = Image.open(str(path_image))
    with open(path_txt, 'r') as file:
      bbox_params = file.read().split('\n')[:-1]
    box_colors = get_box_colors(len(bbox_params))
    cls_idx, count = {}, 0
    for i, params in enumerate(bbox_params):
      cls, x, y, w, h, *ids = params.split(' ')
      if cls not in cls_idx.keys():
        cls_idx[cls] = count; count += 1
      box_params = [x, y, w, h]
      box_params = [float(x) for x in box_params]
      label_str = f"{idx2unit[int(cls)]}"
      for j, id in enumerate(ids):
        idx = j * 10 + int(id)
        if j == 0:
          label_str += f" {id}"
        if id != '0' and j != 0:
          label_str += f" {idx2state[idx]}"
      image = plot_boxes_PIL(
        image, box_params, text=label_str,
        box_color=tuple(box_colors[cls_idx[cls]])
      )
      image = plot_cells_PIL(image, 9, 14)
      # image = plot_cells_PIL(image, 18, 28)
      # image = plot_cells_PIL(image, 36, 56)
    save_image_name = path_image.name
    save_image_name = save_image_name[:-4] + "_label" + save_image_name[-4:]
    # image.save(path_save.joinpath(save_image_name))
    image.show()
    break
