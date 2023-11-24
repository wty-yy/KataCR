# -*- coding: utf-8 -*-
'''
@File    : yolov4_model.py
@Time    : 2023/11/14 09:46:35
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023/11/14: 
'''
from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.yolo.csp_darknet53 import CSPDarkNet, ConvBlock
from katacr.yolo.parser import YOLOArgs

class YOLOBlock(nn.Module):
  filters: int  # 1x1 Conv(filters//2) -> 3x3 Conv(filters)
  conv: nn.Module

  @nn.compact
  def __call__(self, x):
    x = self.conv(filters=self.filters//2, kernel=(1,1))(x)
    x = self.conv(filters=self.filters, kernel=(3,3))(x)
    return x

class SPP(nn.Module):  # Spatial Pyramid Pooling
  @nn.compact
  def __call__(self, x):
    x5 = nn.max_pool(x, (5,5), padding="SAME")
    x9 = nn.max_pool(x, (9,9), padding="SAME")
    x13 = nn.max_pool(x, (13,13), padding="SAME")
    x = jnp.concatenate([x, x5, x9, x13], axis=-1)
    return x

class ScalePredictor(nn.Module):
  conv: nn.Module
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x = self.conv(filters=x.shape[-1]*2, kernel=(3,3))(x)
    x = self.conv(filters=3*(5+self.num_classes), kernel=(1,1), use_norm=False, use_act=False)(x)
    # Shape: [N, 3, H, W, 5 + num_classes]
    return x.reshape((x.shape[0], 3, *x.shape[1:3], 5 + self.num_classes))

class PANet(nn.Module):  # Path Aggregation Network
  num_classes: int
  act: Callable = lambda x: nn.leaky_relu(x, 0.1)

  @nn.compact
  def __call__(self, features, train: bool):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    block = partial(YOLOBlock, conv=conv)
    predictor = partial(ScalePredictor, conv=conv, num_classes=self.num_classes)
    a3, a4, a5 = features  # Resolution: S/8, S/16, S/32

    # Upsampling: b5, b4, b3
    x = block(filters=1024)(a5)
    x = conv(filters=512, kernel=(1,1))(x)
    x = SPP()(x)
    x = block(filters=1024)(x)
    b5 = conv(filters=512, kernel=(1,1))(x)  # Resolution: S/32

    x = conv(filters=256, kernel=(1,1))(b5)
    x = jax.image.resize(x, (x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), 'nearest')
    a4 = conv(filters=256, kernel=(1,1))(a4)
    x = jnp.concatenate([x, a4], axis=-1)
    x = block(filters=512)(x)
    x = block(filters=512)(x)
    b4 = conv(filters=256, kernel=(1,1))(x)  # Resolution: S/16

    x = conv(filters=128, kernel=(1,1))(b4)
    x = jax.image.resize(x, (x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), 'nearest')
    a3 = conv(filters=128, kernel=(1,1))(a3)
    x = jnp.concatenate([x, a3], axis=-1)
    x = block(filters=256)(x)
    x = block(filters=256)(x)
    b3 = conv(filters=128, kernel=(1,1))(x)  # Resolution: S/8

    # Downsampling: o3, o4, o5
    o3 = predictor()(b3)

    x = conv(filters=256, kernel=(3,3), strides=(2,2))(b3)
    x = jnp.concatenate([x, b4], axis=-1)
    x = block(filters=512)(x)
    x = block(filters=512)(x)
    x = conv(filters=256, kernel=(1,1))(x)
    o4 = predictor()(x)

    x = conv(filters=512, kernel=(3,3), strides=(2,2))(x)
    x = jnp.concatenate([x, b5], axis=-1)
    x = block(filters=1024)(x)
    x = block(filters=1024)(x)
    x = conv(filters=512, kernel=(1,1))(x)
    o5 = predictor()(x)

    return [o3, o4, o5]

class YOLOv4(nn.Module):
  num_classes: int

  @nn.compact
  def __call__(self, x, train: bool):
    features = CSPDarkNet(stage_size=[1,2,8,8,4])(x, train)
    outputs = PANet(num_classes=self.num_classes)(features, train)
    return outputs

class TrainState(train_state.TrainState):
  batch_stats: dict

def get_learning_rate_fn(args: YOLOArgs):
    """
    `args.learning_rate`: the target warming up learning rate.
    `args.warmup_epochs`: the epochs get to the target learning rate.
    `args.steps_per_epoch`: number of the steps to each per epoch.
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=args.learning_rate,
        transition_steps=args.warmup_epochs * args.steps_per_epoch
    )
    cosine_epoch = args.total_epochs - args.warmup_epochs
    cosine_fn = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=cosine_epoch * args.steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[args.warmup_epochs * args.steps_per_epoch]
    )
    return schedule_fn

def get_yolov4_state(args: YOLOArgs, verbose=False):
  args.learning_rate_fn = get_learning_rate_fn(args)
  model = YOLOv4(args.num_classes)
  key = jax.random.PRNGKey(args.seed)
  if verbose: print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
  variables = model.init(key, jnp.empty(args.input_shape), train=False)
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    # tx=optax.sgd(learning_rate=args.learning_rate_fn, momentum=args.momentum, nesterov=True),
    tx=optax.adam(learning_rate=args.learning_rate_fn),
    batch_stats=variables['batch_stats']
  )

if __name__ == '__main__':
  from katacr.yolo.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True)
  args.steps_per_epoch = 10
  state = get_yolov4_state(args, verbose=True)
  print(state.params.keys(), state.batch_stats.keys())
  # 'CSPDarkNet_0', 'PANet_0'
