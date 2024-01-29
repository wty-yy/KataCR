from pathlib import Path
import jax.numpy as jnp
from katacr.constants.state_list import num_state_classes
from katacr.constants.dataset import path_dataset

dataset_name = 'ClashRoyale'
path_dataset = Path(path_dataset)
num_classes = 200 + num_state_classes
num_data_workers = 8
train_datasize = 13000

image_shape = (896, 576, 3)  # origin shape = (896, 568, 3)
hsv_h = 0.015  # HSV-Hue augmentation
hsv_s = 0.7  # HSV-Saturation augmentation
hsv_v = 0.4  # HSV-Value augmentation
fliplr = 0.5  # flip left-right (probability)
num_unit = 30  # number of units in one image
intersect_ratio_thre = 0.5  # threshold the intersection ratio

# anchors = jnp.array([
#   [(29.1, 11.3), (18.4, 22.5), (47.7, 16.6)],   # scale: 8
#   [(25.7, 35.3), (68.7, 19.8), (37.9, 43.9)],   # scale: 16
#   [(47.5, 58.3), (62.8, 79.0), (220.6, 37.8)],  # scale: 32
# ], dtype=jnp.float32)
anchors = jnp.array([  # Update: 2024.1.3
  [(57.9, 18.1), (39.2, 42.4), (96.3, 27.8), ],
  [(51.5, 64.5), (141.0, 35.6), (75.1, 77.5), ],
  [(95.4, 101.7), (122.7, 138.9), (402.3, 63.1), ],
], dtype=jnp.float32)

### Training ###
batch_size = 16
total_epochs = 80
coef_box = 0.05
coef_obj = 2.0 * (image_shape[0] / 640) * (image_shape[1] / 640)  # scale image size
coef_cls = 0.5 * 150 / 80  # scale class number
learning_rate_init = 0.01
learning_rate_final = 1e-4
weight_decay = 5e-4
warmup_epochs = 2
momentum = 0.937

if __name__ == '__main__':
  from pathlib import Path
  print(Path(__file__).resolve().parent)