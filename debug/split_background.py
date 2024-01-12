from katacr.build_dataset.generator import cell2pixel
from katacr.build_dataset.generation_config import grid_size
from katacr.utils.detection import plot_cells_PIL
import numpy as np
from PIL import Image

path_background = r'/home/wty/Coding/datasets/CR/images/segment/backgrounds/background25.jpg'
xyxy = np.concatenate([cell2pixel((0,0)), cell2pixel(grid_size)])
img = np.array(Image.open(path_background))[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
img = Image.fromarray(img)
img = plot_cells_PIL(img, *grid_size)
img.show()
