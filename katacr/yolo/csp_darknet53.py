import sys, os
sys.path.append(os.getcwd())

from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *

def mish(x):
  return x * jnp.tanh(jax.nn.softplus(x))

class ConvBlock(nn.Module):
  filters: int
  norm: nn.Module
  act: Callable
  kernel: Tuple[int, int] = (1, 1)
  strides: Tuple[int, int] = (1, 1)
  use_norm: bool = True
  use_act: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.filters, self.kernel, self.strides, use_bias=not self.use_norm)(x)
    if self.use_norm: x = self.norm()(x)
    if self.use_act: x = self.act(x)
    return x

class ResBlock(nn.Module):
  conv: nn.Module
  norm: nn.Module
  keep_channel: bool = True  # When CSP, we need keep same channel in 1x1 Conv.

  @nn.compact
  def __call__(self, x):
    n = x.shape[-1] // 2
    residue = x
    x = self.conv(filters=2*n if self.keep_channel else n, kernel=(1,1))(x)
    x = self.conv(filters=2*n, kernel=(3,3))(x)
    return x + residue

class CSPDarkNet(nn.Module):
  stage_size: Sequence[int]
  act: Callable = mish

  @nn.compact
  def __call__(self, x, train: bool):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    block = partial(ResBlock, conv=conv, norm=norm)
    x = conv(filters=32, kernel=(3,3))(x)
    outputs = []
    for i, block_size in enumerate(self.stage_size):
      x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=(2,2))(x)
      if i == 0:  # first stage don't need CSP
        assert(block_size == 1)
        x = block(keep_channel=False)(x)
      else:  # CSP with block_size * ResBlock
        route = conv(filters=x.shape[-1]//2, kernel=(1,1))(x)
        x = conv(filters=x.shape[-1]//2, kernel=(1,1))(x)
        for _ in range(block_size):
          x = block()(x)
        x = conv(filters=x.shape[-1], kernel=(1,1))(x)
        x = jnp.concatenate([x, route], axis=-1)
        x = conv(filters=x.shape[-1], kernel=(1,1))(x)  # fusion
      if i > 1: outputs.append(x)
    return outputs

class PreTrain(nn.Module):
  darknet: nn.Module

  @nn.compact
  def __call__(self, x, train: bool):
    x = self.darknet(x, train)[-1]
    x = jnp.mean(x, (1, 2))
    x = nn.Dense(1000)(x)
    return x

if __name__ == '__main__':
  pass
