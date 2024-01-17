from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.detection.new_csp_darknet53 import CSPDarkNet, ConvBlock, CSP
from katacr.detection.parser import YOLOv5Args
from katacr.detection.train_state import TrainState, zeros_grads

class SPP(nn.Module):  # Spatial Pyramid Pooling(F), same result but faster x2.5
  conv: nn.Module
  @nn.compact
  def __call__(self, x):
    n = x.shape[-1]
    x = self.conv(filters=n//2, kernel=(1,1))(x)
    y1 = nn.max_pool(x, (5, 5), padding='SAME')
    y2 = nn.max_pool(y1, (5, 5), padding='SAME')
    y3 = nn.max_pool(y2, (5, 5), padding='SAME')
    x = jnp.concatenate([x, y1, y2, y3], axis=-1)
    x = self.conv(filters=n, kernel=(1,1))(x)
    return x

class ScalePredictor(nn.Module):
  conv: nn.Module
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x = self.conv(filters=3*(5+self.num_classes), kernel=(1,1), use_norm=False, use_act=False)(x)
    # Shape: [N, 3, H, W, 5 + num_classes]
    return x.reshape((*x.shape[:3], 3, 5 + self.num_classes)).transpose((0, 3, 1, 2, 4))

class PANet(nn.Module):  # Path Aggregation Network
  num_classes: int
  act: Callable = nn.silu

  @nn.compact
  def __call__(self, features, train: bool):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    spp = partial(SPP, conv=conv)
    csp = partial(CSP, n_bottleneck=3, conv=conv, shortcut=False)
    def upsample(x):
      return jax.image.resize(x, (x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), 'nearest')

    predictor = partial(ScalePredictor, conv=conv, num_classes=self.num_classes)
    a3, a4, a5 = features  # S/8, S/16, S/32

    # Upsampling: b5, b4, b3
    x = spp()(a5)
    b5 = conv(filters=512, kernel=(1,1))(x)  # S/32
    x = upsample(b5)
    x = jnp.concatenate([x, a4], axis=-1)
    x = csp(output_channel=512)(x)
    b4 = conv(filters=256, kernel=(1,1))(x)  # S/16
    x = upsample(b4)
    x = jnp.concatenate([x, a3], axis=-1)
    b3 = csp(output_channel=256)(x)  # S/8

    # Downsampling: o3, o4, o5
    o3 = predictor()(b3)
    x = conv(filters=256, kernel=(3,3), strides=(2,2))(b3)
    x = jnp.concatenate([x, b4], axis=-1)
    x = csp(output_channel=512)(x)
    o4 = predictor()(x)
    x = conv(filters=512, kernel=(3,3), strides=(2,2))(x)
    x = jnp.concatenate([x, b5], axis=-1)
    x = csp(output_channel=1024)(x)
    o5 = predictor()(x)
    return [o3, o4, o5]

class YOLOv5(nn.Module):
  num_classes: int

  @nn.compact
  def __call__(self, x, train: bool):
    features = CSPDarkNet()(x, train)
    outputs = PANet(num_classes=self.num_classes)(features, train)
    return outputs

def get_learning_rate_fn(args: YOLOv5Args, init_value=0.0):
  """
  `args.learning_rate`: the target warming up learning rate.
  `args.warmup_epochs`: the epochs get to the target learning rate.
  `args.steps_per_epoch`: number of the steps to each per epoch.
  """
  warmup_fn = optax.linear_schedule(
    init_value=init_value,  # bias start from 0.1, other start from 0.0
    end_value=args.learning_rate,
    transition_steps=args.warmup_epochs * args.steps_per_epoch
  )
  second_epoch = args.total_epochs - args.warmup_epochs
  if args.use_cosine_decay:
    second_fn = optax.cosine_decay_schedule(
      init_value=args.learning_rate,
      decay_steps=second_epoch * args.steps_per_epoch,
      alpha=args.learning_rate_final / args.learning_rate
    )
  else:
    second_fn = optax.linear_schedule(
      init_value=args.learning_rate,
      end_value=args.learning_rate_final,
      transition_steps=second_epoch * args.steps_per_epoch
    )
  schedule_fn = optax.join_schedules(
    schedules=[warmup_fn, second_fn],
    boundaries=[args.warmup_epochs * args.steps_per_epoch]
  )
  return schedule_fn

def get_state(args: YOLOv5Args, verbose=False):
  args.learning_rate_fn = get_learning_rate_fn(args)
  args.learning_rate_bias_fn = get_learning_rate_fn(args, init_value=0.1)
  model = YOLOv5(args.num_classes)
  key = jax.random.PRNGKey(args.seed)
  if verbose: print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
  variables = model.init(key, jnp.empty(args.input_shape), train=False)
  decay_mask = jax.tree_map(lambda x: x.ndim > 1, variables['params'])
  state = TrainState.create(
    apply_fn=model.apply,
    params=variables.get('params'),
    tx=optax.chain(
      optax.clip_by_global_norm(max_norm=10.0),
      optax.add_decayed_weights(weight_decay=args.weight_decay, mask=decay_mask),
      optax.sgd(learning_rate=args.learning_rate_fn, momentum=args.momentum, nesterov=True)
    ),
    tx_bias=optax.chain(
      optax.clip_by_global_norm(max_norm=10.0),
      optax.add_decayed_weights(weight_decay=args.weight_decay, mask=decay_mask),
      optax.sgd(learning_rate=args.learning_rate_bias_fn, momentum=args.momentum, nesterov=True)
    ),
    batch_stats=variables.get('batch_stats'),
    grads=variables.get('params'),
    accumulate=args.accumulate,
    acc_count=0,
    ema=variables
  )
  for i in range(3):
    s = 2 ** (i + 3)
    bias = state.params['PANet_0'][f'ScalePredictor_{i}']['ConvBlock_0']['Conv_0']['bias']
    bias = bias.reshape(3, -1)
    bias = bias.at[:, 4].set(jnp.log(8 / (args.image_shape[0] / s) * (args.image_shape[1] / s)))  # assume 8 target boxes per layer
    bias = bias.at[:, 5:].set(jnp.log(0.6 / (args.num_classes - 1 + 1e-6)))  # the distribution for each class in dataset
    bias = bias.reshape(-1)
    state.params['PANet_0'][f'ScalePredictor_{i}']['ConvBlock_0']['Conv_0']['bias'] = bias
  state = zeros_grads(state)
  return state

if __name__ == '__main__':
  from katacr.detection.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True)
  state = get_state(args, verbose=True, use_init=False)
  # print(state.params.keys(), state.batch_stats.keys())
  # 'CSPDarkNet_0', 'PANet_0'
