from katacr.build_dataset.generator import cell2pixel
from katacr.build_dataset.generation_config import grid_size, background_size
from katacr.utils.detection import plot_cells_PIL
from katacr.build_dataset.constant import path_logs
import numpy as np
from PIL import Image

path_save = path_logs / "background_cells"
path_save.mkdir(exist_ok=True)
# path_background = r'/home/wty/Coding/datasets/CR/images/segment/backgrounds/background25.jpg'
path_background = r'/home/wty/Coding/datasets/CR/images/part2/background/background26.jpg'
xyxy = np.concatenate([cell2pixel((0,0)), cell2pixel(grid_size)])
img = np.array(Image.open(path_background).resize(background_size))[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
img = Image.fromarray(img)
img = plot_cells_PIL(img, *grid_size)
# img = plot_cells_PIL(img, grid_size[0], grid_size[1]+1)
img.save(path_save / path_background.rsplit('/',1)[-1])
img.show()
