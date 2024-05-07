import jax, jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple
from functools import partial
from katacr.utils import Config
import numpy as np

class CSPDarkNetConfig(Config):
  stage_size = [1, 1, 2]  # [3, 6, 9, 3]
  filters = 8  # 64

class ConvBlock(nn.Module):
  filters: int
  act: Callable
  kernel: Tuple[int, int] = (1, 1)
  strides: Tuple[int, int] = (1, 1)
  padding: str | Tuple[int, int] = 'SAME'

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.filters, self.kernel, self.strides, self.padding)(x)
    x = self.act(x)
    return x

class BottleNeck(nn.Module):
  conv: nn.Module
  shortcut: bool

  @nn.compact
  def __call__(self, x):
    residue = x
    c = x.shape[-1]
    x = self.conv(filters=c, kernel=(1,1))(x)
    x = self.conv(filters=c, kernel=(3,3))(x)
    if self.shortcut:
      x = x + residue
    return x

class CSP(nn.Module):
  n_bottleneck: int
  conv: nn.Module
  output_channel: int
  shortcut: bool = True

  @nn.compact
  def __call__(self, x):
    neck = partial(BottleNeck, conv=self.conv, shortcut=self.shortcut)
    n = self.output_channel // 2
    route = self.conv(filters=n, kernel=(1,1))(x)
    x = self.conv(filters=n, kernel=(1,1))(x)
    for _ in range(self.n_bottleneck):
      x = neck()(x)
    x = jnp.concatenate([x, route], axis=-1)
    return self.conv(filters=self.output_channel, kernel=(1,1))(x)

class CSPDarkNet(nn.Module):
  cfg: CSPDarkNetConfig
  act: Callable = nn.silu

  @nn.compact
  def __call__(self, x):
    conv = partial(ConvBlock, act=self.act)
    csp = partial(CSP, conv=conv)
    x = conv(filters=self.cfg.filters, kernel=(6,6), strides=(2,2), padding=(2,2))(x)  # P1
    for i, n_blockneck in enumerate(self.cfg.stage_size):  # start from P2
      if i != 0:
        x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=(2,2))(x)
      x = csp(n_bottleneck=n_blockneck, output_channel=x.shape[-1])(x)
    return x

if __name__ == '__main__':
  # cfg = CSPDarkNetConfig(stage_size=[2,3,5], filters=8)  # 24,168 (96.7 KB)
  cfg = CSPDarkNetConfig(stage_size=[3,4,5], filters=32)  # 407,552 (1.6 MB)
  model = CSPDarkNet(cfg)
  # x = np.empty((32, 30, 8, 24, 1))
  x = np.empty((32, 30, 32, 18, 15))
  print(model.tabulate(jax.random.PRNGKey(42), x))
  print(x.mean(-1).shape)

