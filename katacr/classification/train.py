import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Callable, Tuple, Any, Sequence
from functools import partial
from katacr.utils import Config
from katacr.constants.card_list import card_list
import numpy as np

class ModelConfig(Config):
  stage_sizes = [3, 3, 3]
  filters = 16
  num_class = len(card_list)

class TrainConfig(Config):
  lr_fn: Callable
  steps_per_epoch: int
  image_size = (64, 80)
  seed = 42
  weight_decay = 1e-4
  lr = 0.01
  total_epochs = 10
  warmup_epochs = 1
  batch_size = 32
  betas = (0.9, 0.999)
  ### Dataset ###
  num_workers = 4
  ### Augmentation ###
  h_hsv = 0.015
  s_hsv = 0.7
  v_hsv = 0.4
  rotate = 10
  scale = 0.3
  translate = 0.05

class TrainState(train_state.TrainState):
  batch_stats: dict

ModuleDef = Any
class BottleneckResNetBlock(nn.Module):
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.act(self.norm()(y))
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.act(self.norm()(y))
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)
    if residual.shape != y.shape:
      residual = self.conv(
        self.filters * 4, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)
    return self.act(y + residual)

class ResNet(nn.Module):
  cfg: ModelConfig

  @nn.compact
  def __call__(self, x, train = True):
    conv = partial(nn.Conv, use_bias=False)
    norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5)
    x = conv(self.cfg.filters, (3, 3), strides=(2, 2), name='conv_init')(x)
    x = nn.relu(norm(name='bn_init')(x))
    for i, block_size in enumerate(self.cfg.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = BottleneckResNetBlock(
          self.cfg.filters * 2 ** i,
          strides=strides,
          conv=conv,
          norm=norm,
          act=nn.relu
        )(x)
    x = jnp.mean(x, (1, 2))
    x = nn.Dense(self.cfg.num_class, use_bias=False)(x)
    return x
  
  def create_fns(self):
    def model_step(state: TrainState, x, y, train: bool):
      def loss_fn(params):
        logits, updates = state.apply_fn(
          {'params': params, 'batch_stats': state.batch_stats},
          x, train=train,
          mutable=['batch_stats'],
        )
        loss = -(jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), y]).mean()
        acc = (jnp.argmax(logits, -1).reshape(-1) == y.reshape(-1)).mean()
        return loss, (updates, acc)
      (loss, (updates, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(batch_stats=updates['batch_stats'])
      return state, (loss, acc)
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, x):
      logits = state.apply_fn({'params': state.params}, x, train=False)
      pred = jax.nn.softmax(logits)
      return pred
    self.predict = jax.jit(predict)
  
  def get_states(self, train_cfg: TrainConfig, verbose=False, train=True) -> TrainState:
    def lr_fn():
      warmup_steps = train_cfg.warmup_epochs * train_cfg.steps_per_epoch
      warmup_fn = optax.linear_schedule(0.0, train_cfg.lr, warmup_steps)
      second_steps = max(train_cfg.total_epochs * train_cfg.steps_per_epoch - warmup_steps, 1)
      second_fn = optax.cosine_decay_schedule(
        train_cfg.lr, second_steps, 0.01
      )
      return optax.join_schedules(
        schedules=[warmup_fn, second_fn],
        boundaries=[warmup_steps]
      )
    verbose_rng, init_rng = jax.random.split(jax.random.PRNGKey(train_cfg.seed), 2)
    if not train:
      return TrainState.create(apply_fn=self.apply, params={'a': 1}, tx=optax.sgd(1))
    dummy = jnp.zeros((train_cfg.batch_size, *train_cfg.image_size[::-1], 3))
    if verbose: print(self.tabulate(verbose_rng, dummy, train=False))
    variables = self.init(init_rng, dummy, train=False)
    print("Model parameters:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variables)[0]]))
    wd_mask = jax.tree_util.tree_map(lambda x: x.ndim > 1, variables['params'])
    train_cfg.lr_fn = lr_fn()
    state = TrainState.create(
       apply_fn=self.apply,
       params=variables['params'],
       tx=optax.adamw(train_cfg.lr_fn, train_cfg.betas[0], train_cfg.betas[1], weight_decay=train_cfg.weight_decay, mask=wd_mask),
       batch_stats=variables['batch_stats'],
    )
    return state

from torch.utils.data import DataLoader, Dataset
from katacr.constants.card_list import card_list
from katacr.utils.detection.data import transform_affine, transform_hsv, transform_resize_and_pad
import cv2
class CardDataset(Dataset):
  def __init__(self, path_dataset, mode='train', repeat=20, cfg: TrainConfig = None):
    self.path_dataset, self.mode, self.repeat, self.cfg = path_dataset, mode, repeat, cfg
    self.idx2card = dict(enumerate(card_list))
    self.card2idx = {c: i for i, c in enumerate(card_list)}
    self.paths = sorted(Path(self.path_dataset).glob('*.jpg'))
    self.images, self.labels = [], []
    for p in self.paths:
      self.images.append(cv2.imread(str(p)))
      self.labels.append(self.card2idx[p.stem])
  
  def __len__(self):
    return len(self.images) * (1 if self.mode == 'val' else self.repeat)
  
  def __getitem__(self, idx):
    idx = idx % len(self.images)
    img = self.images[idx].copy()
    cfg = self.cfg
    if self.mode == 'train':
      img = transform_affine(img, rot=cfg.rotate, scale=cfg.scale, translate=cfg.translate)
      img = transform_hsv(img, cfg.h_hsv, cfg.s_hsv, cfg.v_hsv)
    img, _ = transform_resize_and_pad(img, cfg.image_size[::-1])
    img = np.ascontiguousarray(img[..., ::-1])
    return img, self.labels[idx]

class DatasetBuilder:
  def __init__(self, path_dataset):
    self.path_dataset = path_dataset
  
  def get_dataloader(self, train_cfg: TrainConfig, mode='train'):
    return DataLoader(
      CardDataset(self.path_dataset, mode=mode, repeat=1 if mode=='val' else 20, cfg=train_cfg),
      batch_size=train_cfg.batch_size,
      shuffle=mode=='train',
      num_workers=train_cfg.num_workers,
      persistent_workers=train_cfg.num_workers > 0,
      drop_last=True,
    )

from katacr.utils.logs import Logs, MeanMetric
logs = Logs(
    init_logs={
      'train_loss': MeanMetric(),
      'train_acc': MeanMetric(),
      'val_loss': MeanMetric(),
      'val_acc': MeanMetric(),
      'epoch': MeanMetric(),
      'SPS': MeanMetric(),
      'learning_rate': MeanMetric(),
    },
    folder2name={
      'Charts': ['SPS', 'epoch', 'learning_rate'],
      'Metrics': ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
    }
)

def get_args_and_writer():
    from katacr.utils.parser import Parser, datetime
    parser = Parser(model_name="CardClassification", wandb_project_name="ClashRoyale Card")
    parser.add_argument("--image-size", type=tuple, default=(64, 80))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.get_args()
    args.lr = args.learning_rate
    args.run_name = f"{args.model_name}__lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    writer = parser.get_writer(args)
    return args, writer

from katacr.build_dataset.constant import path_dataset
from tqdm import tqdm
def train():
  ### Parse Augmentations and Get WandB Writer ###
  args, writer = get_args_and_writer()
  ### Dataset ###
  ds_builder = DatasetBuilder(str(path_dataset / "images/card_classification"))
  train_cfg = TrainConfig(**vars(args))
  model_cfg = ModelConfig(**vars(args))
  train_ds = ds_builder.get_dataloader(train_cfg, mode='train')
  val_ds = ds_builder.get_dataloader(train_cfg, mode='val')
  args.steps_per_epoch = train_cfg.steps_per_epoch = len(train_ds)
  ### Build Model ###
  model = ResNet(model_cfg)
  model.create_fns()
  state = model.get_states(train_cfg, verbose=True)
  ### Build Checkpoint Manager ###
  from katacr.utils.ckpt_manager import CheckpointManager
  ckpt_manager = CheckpointManager(args.path_cp)
  ### Train and Evaulate ###
  for ep in range(train_cfg.total_epochs):
    print(f"Epoch: {ep+1}/{train_cfg.total_epochs}")
    print("Training...")
    logs.reset()
    bar = tqdm(train_ds, ncols=80)
    for x, y in bar:
      x, y = x.numpy().astype(np.float32) / 255., y.numpy().astype(np.int32)
      state, (loss, acc) = model.model_step(state, x, y, train=True)
      logs.update(['train_loss', 'train_acc'], [loss, acc])
      bar.set_description(f"loss={loss:.4f}, acc={acc:.4f}")
      if state.step % 10 == 0:
        logs.update(
          ['SPS', 'epoch', 'learning_rate'],
          [10 / logs.get_time_length(), ep+1, train_cfg.lr_fn(state.step)]
        )
        logs.writer_tensorboard(writer, state.step)
        logs.reset()
    logs.reset()
    print("Evaluating...")
    bar = tqdm(val_ds, ncols=80)
    for x, y in bar:
      x, y = x.numpy().astype(np.float32) / 255., y.numpy().astype(np.int32)
      state, (loss, acc) = model.model_step(state, x, y, train=False)
      logs.update(['val_loss', 'val_acc'], [loss, acc])
    logs.writer_tensorboard(writer, state.step)
    ckpt_manager.save(ep+1, state, vars(args))
  ckpt_manager.close()
  writer.close()
  if args.wandb_track:
    import wandb
    wandb.finish()

if __name__ == '__main__':
  train()
