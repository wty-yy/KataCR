import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Callable, Tuple, Any
from functools import partial
from katacr.utils import Config
import numpy as np

class ResNetConfig(Config):  # No BN ResNet  # ResNet50
  stage_sizes = [3, 4, 6]                    # (3, 4, 6, 3)
  filters = 16                               # 64

ModuleDef = Any
class BottleneckResNetBlock(nn.Module):
  filters: int
  conv: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.act(y)
    y = self.conv(self.filters * 2, (1, 1))(y)
    if residual.shape != y.shape:
      residual = self.conv(
        self.filters * 2, (1, 1), self.strides, name='conv_proj'
      )(residual)
    return self.act(y + residual)

class ResNet(nn.Module):
  cfg: ResNetConfig

  @nn.compact
  def __call__(self, x):
    conv = nn.Conv
    x = conv(self.cfg.filters, (3, 3), strides=(2, 2), name='conv_init')(x)
    x = nn.relu(x)
    for i, block_size in enumerate(self.cfg.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = BottleneckResNetBlock(
          self.cfg.filters * 2 ** i,
          strides=strides,
          conv=conv,
          act=nn.relu
        )(x)
    return x

if __name__ == '__main__':
  # cfg = ResNetConfig()
  cfg = ResNetConfig(stage_size=[1,1,2], filters=4)  # 24,912 (99.6 KB)
  cfg = ResNetConfig(stage_size=[3,4,6], filters=16)  # 392,736 (1.6 MB)
  model = ResNet(cfg)
  # x = np.empty((32, 30, 8, 24, 1))
  x = np.empty((32, 30, 32, 18, 15))
  print(model.tabulate(jax.random.PRNGKey(42), x))
  print(x.mean(-1).shape)

