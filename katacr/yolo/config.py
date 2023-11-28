from pathlib import Path
import jax.numpy as jnp
from katacr.constants.state_list import num_state_classes

# path_darknet_weights = Path("/home/yy/Coding/models/YOLOv4/CSPDarkNet53-0050-lite")

dataset_name = 'ClashRoyale'
# path_dataset = Path("/home/wty/Coding/datasets/CR")
path_dataset = Path("/home/yy/Coding/datasets/CR")
num_classes = 200 + num_state_classes  # 200 is the maximum unit classes number
num_data_workers = 8
repeat = 10

# image_shape = (896, 568, 3)  # origin Image
image_shape = (512, 320, 3)  # model input with /2 scale, two image input (current image, last frames)
anchors = jnp.array([
  [(29.1, 11.3), (18.4, 22.5), (47.7, 16.6)],   # scale: 8
  [(25.7, 35.3), (68.7, 19.8), (37.9, 43.9)],   # scale: 16
  [(47.5, 58.3), (62.8, 79.0), (220.6, 37.8)],  # scale: 32
], dtype=jnp.float32)

### Training ###
batch_size = 16
total_epochs = 100
coef_noobj = 2.0
coef_coord = 2.0
coef_obj = 1.0
coef_class = 1.0
# base_learning_rate = 2.5e-4
# base_learning_rate = 1e-3
# learning_rate = base_learning_rate * batch_size / 256
learning_rate = 1e-4
weight_decay = 1e-4
warmup_epochs = 2
momentum = 0.9  # if optimizer is SGD