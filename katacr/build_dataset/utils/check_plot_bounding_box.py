from katacr.utils.related_pkgs.utility import *
from katacr.utils.detection import plot_box_PIL, plot_cells_PIL, get_box_colors, build_label2colors
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.constants.state_list import idx2state
from katacr.constants.label_list import idx2unit
from katacr.build_dataset.constant import path_logs
from katacr.detection.cfg import image_shape
from PIL import Image

if __name__ == '__main__':
  path_manager = PathManager()
  # paths = path_manager.search(subset='images', part=2, regex=r'^\d+.txt')
  paths = path_manager.search(subset='images', part=2, name="OYASSU_20230305_episodes/4/", regex=r'04875.txt')
  # paths = path_manager.search(subset='images', part=2, regex=r'^04275.txt')
  path_save = path_logs / "label_images"
  path_save.mkdir(exist_ok=True)
  fliplr = False
  for path_txt in tqdm(paths):
    path_image = path_txt.parent / (path_txt.name.rsplit('.',1)[0]+".jpg")
    image = Image.open(str(path_image))
    box = np.loadtxt(path_txt)
    print(box.shape)
    # with open(path_txt, 'r') as file:
    #   bbox_params = file.read().split('\n')[:-1]
    image = np.array(image.resize(image_shape[:2][::-1]))
    if fliplr:
      image = np.fliplr(image)
      box[:, 1] = 1 - box[:, 1]
    image = Image.fromarray(image.astype('uint8'))
    colors = build_label2colors(box[:,0])
    for i, params in enumerate(box):
      cls, x, y, w, h, *idxs = params
      b = [x, y, w, h]
      label_str = f"{idx2unit[int(cls)]}"
      for j, idx in enumerate(idxs):
        if j == 0:
          label_str += f" {int(idx)}"
        if idx != 0 and j != 0:
          label_str += f" {idx2state[j*10+int(idx)]}"
      image = plot_box_PIL(
        image, b, text=label_str,
        box_color=tuple(colors[cls])
      )
    # image = plot_cells_PIL(image, 9, 14)  # scale 64
    # image = plot_cells_PIL(image, 18, 28)  # scale 32
    # image = plot_cells_PIL(image, 36, 56)  # scale 16
    # image = plot_cells_PIL(image, 72, 112)  # scale 8
    # scale = 8
    # cw = int(image.size[0]/scale)
    # ch = int(image.size[1]/scale)
    # print("Cell:", cw, ch)
    # image = plot_cells_PIL(image, cw, ch)
    save_image_name = path_image.name
    save_image_name = save_image_name[:-4] + "_label" + ("_fliplr" if fliplr else "") + save_image_name[-4:]
    image.save(path_save.joinpath(save_image_name))
    image.show()
    break
