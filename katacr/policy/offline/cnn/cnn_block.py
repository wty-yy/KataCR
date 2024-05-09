import jax
import flax.linen as nn
from katacr.utils import Config
import numpy as np

class CNNBlockConfig(Config):
  filters = [32, 64, 64]
  kernels = [8, 4, 3]
  strides = [4, 2, 1]

class CNNBlock(nn.Module):
  cfg: CNNBlockConfig

  @nn.compact
  def __call__(self, x):
    for f, k, s in zip(self.cfg.filters, self.cfg.kernels, self.cfg.strides):
      x = nn.relu(nn.Conv(f, (k, k), s, padding='SAME')(x))
    return x

if __name__ == '__main__':
  # cfg = CNNBlockConfig(filter=[32, 64, 64], kernels=[8, 4, 3], strides=[4, 2, 1])
  cfg = CNNBlockConfig(filters=[16, 32, 32], kernels=[6, 3, 3], strides=[2, 2, 2])  # (RGB) Total Parameters: 15,632 (62.5 KB), (GRAY) Total Parameters: 14,480 (57.9 KB)
  # cfg = CNNBlockConfig(filters=[64, 128, 128], kernels=[6, 3, 3], strides=[2, 2, 2])  # Total Parameters: 256,064 (1.0 MB)
  model = CNNBlock(cfg)
  # x = np.empty((32, 30, 84, 84, 4))
  x = np.empty((32, 30, 8, 24, 3))
  # x = np.empty((32, 30, 32, 18, 15))
  print(model.tabulate(jax.random.PRNGKey(42), x))
  print(x.mean(-1).shape)

