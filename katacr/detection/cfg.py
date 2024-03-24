from pathlib import Path
import jax.numpy as jnp
from katacr.constants.state_list import num_state_classes
from katacr.constants.dataset import path_dataset

dataset_name = 'ClashRoyale'
path_dataset = Path(path_dataset)
num_classes = 200 + num_state_classes
num_data_workers = 8
train_datasize = 100000 # 13000 -> 100000

image_shape = (896, 576, 3)  # origin shape = (896, 568, 3)
# hsv_h = 0.015  # HSV-Hue augmentation
# hsv_s = 0.7  # HSV-Saturation augmentation
# hsv_v = 0.4  # HSV-Value augmentation
fliplr = 0.5  # flip left-right (probability)
num_unit = 40  # number of units in one image
intersect_ratio_thre = 0.5  # threshold the intersection ratio
generation_map_mode = 'naive'  # The mode of update the prob map, dynamic or naive

# anchors = jnp.array([  # Update: 2024.1.3, v0.4.4 previous
#   [(57.9, 18.1), (39.2, 42.4), (96.3, 27.8), ],
#   [(51.5, 64.5), (141.0, 35.6), (75.1, 77.5), ],
#   [(95.4, 101.7), (122.7, 138.9), (402.3, 63.1), ],
# ], dtype=jnp.float32)
# anchors = jnp.array([  # Update: 2024.02.08, v0.4.5, v0.4.5.6 previous
#   [(59.3, 20.8), (41.8, 46.0), (107.8, 32.9), ],
#   [(61.7, 65.6), (87.9, 88.9), (129.5, 135.4), ],
#   [(379.5, 61.5), (204.7, 206.0), (331.7, 272.2), ],
# ], dtype=jnp.float32)
anchors = jnp.array([  # Update: 2024.03.10, v0.5, n=4431
  [(34.6, 29.6), (47.5, 52.4), (61.3, 77.0), ],
  [(84.1, 61.3), (138.5, 44.1), (90.7, 94.5), ],
  [(119.8, 133.7), (236.1, 189.7), (141.1, 371.9), ],
], dtype=jnp.float32)

### Training ###
batch_size = 16
total_epochs = 150
coef_box = 0.05
coef_obj = 2.0 * (image_shape[0] / 640) * (image_shape[1] / 640)  # scale image size
coef_cls = 0.5 * 150 / 80  # scale class number
learning_rate_init = 0.01
learning_rate_final = 1e-4
weight_decay = 5e-4
warmup_epochs = 5
momentum = 0.937

if __name__ == '__main__':
  from pathlib import Path
  print(Path(__file__).resolve().parent)