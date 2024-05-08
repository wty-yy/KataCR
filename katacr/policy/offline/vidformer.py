"""
Use ViD feature to s in DT, remove auto position embedding
"""
import jax, jax.numpy as jnp
import flax.linen as nn
import flax, optax
import numpy as np
from katacr.policy.offline.train_state import TrainState, accumulate_grads
from typing import Callable, Sequence
from katacr.utils import Config
from functools import partial
from einops import rearrange
from katacr.policy.offline.dataset import BAR_SIZE
from katacr.policy.offline.cnn.csp_darknet import CSPDarkNet, CSPDarkNetConfig

Dense = partial(nn.Dense, kernel_init=nn.initializers.normal(stddev=0.02))
Embed = partial(nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))

class ViDConfig(Config):
  n_embd_global = 192
  n_head_global = 8
  patch_size = (2, 2)
  n_embd_local = 64
  n_head_local = 4
  n_block = 6  # ViDBlock number
  p_drop_embd = 0.1
  p_drop_resid = 0.1
  p_drop_attn = 0.1
  bar_size = BAR_SIZE
  bar_cfg = CSPDarkNetConfig(stage_sizes=[2,3,5], filters=8)
  n_elixir = 10

  def __init__(self, n_unit, n_cards, n_step, max_timestep, **kwargs):
    self.n_unit, self.n_cards, self.n_step = n_unit, n_cards, n_step
    self.max_timestep = max_timestep
    for k, v in kwargs.items():
      setattr(self, k, v)
    self.n_bar_size = np.prod(self.bar_size)
    assert self.n_embd_global % self.n_head_global == 0, "n_embd_global must be devided by n_head_global"
    assert self.n_embd_local % self.n_head_local == 0, "n_embd_local must be devided by n_head_local"

class TrainConfig(Config):
  seed = 42
  weight_decay = 0.1
  lr = 6e-4
  total_epochs = 10
  batch_size = 16
  betas = (0.9, 0.95)  # Adamw beta1, beta2
  warmup_tokens = 512*20  # 375e6
  clip_global_norm = 1.0
  lr_fn: Callable

  def __init__(self, steps_per_epoch, n_step, accumulate, **kwargs):
    self.steps_per_epoch = steps_per_epoch
    self.n_step = n_step
    self.accumulate = accumulate
    for k, v in kwargs.items():
      setattr(self, k, v)

class CausalSelfAttention(nn.Module):
  n_embd: int  # NOTE: n_embd % n_head == 0
  n_head: int
  cfg: ViDConfig

  @nn.compact
  def __call__(self, q, k = None, v = None, mask = None, train = True):
    assert q is not None, "The q must not be None"
    if k is None and v is None: k = v = q
    elif v is None: v = k
    D = self.n_embd // self.n_head  # hidden dim
    B, L, _ = q.shape  # Bachsize, token length, embedding dim
    if mask is not None:
      if mask.ndim == 2: mask = rearrange(mask, 'h w -> 1 1 h w')
      elif mask.ndim == 3: mask = rearrange(mask, 'b h w -> b 1 h w')
    q = Dense(self.n_embd)(q).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    k = Dense(self.n_embd)(k).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    v = Dense(self.n_embd)(v).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    attn = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(D)
    if mask is not None:
      attn = jnp.where(mask == 0, -1e18, attn)
    attn = jax.nn.softmax(attn)
    attn = nn.Dropout(self.cfg.p_drop_attn)(attn, deterministic=not train)
    y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.n_embd)
    y = Dense(self.n_embd)(y)
    y = nn.Dropout(self.cfg.p_drop_resid)(y, deterministic=not train)
    return y

class TransformerBlock(nn.Module):
  n_embd: int
  n_head: int
  cfg: ViDConfig

  @nn.compact
  def __call__(self, x, mask = None, train = True):
    csa = CausalSelfAttention(self.n_embd, self.n_head, self.cfg)
    x = x + csa(nn.LayerNorm()(x), mask=mask, train=train)
    dropout = nn.Dropout(self.cfg.p_drop_resid)
    mlp = nn.Sequential([
      Dense(4*self.n_embd), nn.gelu,
      Dense(self.n_embd)
    ])
    x = x + dropout(mlp(nn.LayerNorm()(x)), deterministic=not train)
    return x

class ViDBlock(nn.Module):
  cfg: ViDConfig

  @nn.compact
  def __call__(self, xl, xg, train = True):
    local_block = TransformerBlock(n_embd=self.cfg.n_embd_local, n_head=self.cfg.n_head_local, cfg=self.cfg)
    global_block = TransformerBlock(n_embd=self.cfg.n_embd_global, n_head=self.cfg.n_head_global, cfg=self.cfg)
    B, N, M, nl = xl.shape  # Batch, Step Length, Group Token Length, n_embd_local
    B, N3, ng = xg.shape  # Batch, Step Length * 3, n_embd_global
    xl = local_block(xl.reshape(B * N, M, nl), train=train).reshape(B, N, M, nl)
    zg = Dense(ng)(xl.reshape(B, N, M * nl)).reshape(B, N, 1, ng)  # Local state to globale state
    xg = xg.reshape(B, N, 3, ng)  # a1 a2 r
    zg = jnp.concatenate([xg, zg], 2).reshape(B, N3+N, ng)  # a1 a2 r s
    mask = jnp.tri(N3 + N)
    mask = mask.at[jnp.arange(N) * 4, jnp.arange(N) * 4 + 1].set(1)  # action1 can see action2
    zg = global_block(zg, mask=mask, train=train)
    return xl, zg

class ViDformer(nn.Module):
  cfg: ViDConfig

  @nn.compact
  def __call__(self, s, a, r, timestep, train = True):
    cfg = self.cfg
    nl, ng = cfg.n_embd_local, cfg.n_embd_global
    arena, arena_mask, cards, elixir = s['arena'], s['arena_mask'], s['cards'], s['elixir']
    ### Create Arena Feature ###
    B, N, H, W = arena.shape[:-1]  # Batch, Step Length, Height, Width
    cls, bel = arena[...,0], arena[...,1]
    cls = Embed(cfg.n_unit+1, 8)(cls)  # (B, N, H, W, 8)
    bel = bel[..., None]  # (B, N, H, W, 1)
    bar1 = arena[...,-2*cfg.n_bar_size:-cfg.n_bar_size].reshape(B, N, H, W, cfg.bar_size[1], cfg.bar_size[0], 1)
    bar1 = jnp.array(bar1, np.float32) / 255.
    bar2 = arena[...,-cfg.n_bar_size:].reshape(B, N, H, W, cfg.bar_size[1], cfg.bar_size[0], 1)
    bar2 = jnp.array(bar2, np.float32) / 255.
    bar1 = CSPDarkNet(cfg.bar_cfg)(bar1).mean(-1)[...,0,:]  # (B, N, H, W, 3)
    bar2 = CSPDarkNet(cfg.bar_cfg)(bar2).mean(-1)[...,0,:]  # (B, N, H, W, 3)
    arena = jnp.concatenate([cls, bel, bar1, bar2], -1)  # (B, N, H, W, 15)
    arena = arena * arena_mask[..., None]  # add mask
    ### Embedding Global Token ###
    # pos_embd = nn.Embed(4*N, ng, embedding_init=nn.initializers.zeros)(jnp.arange(4*N))  # (1, N, Ng) Don't need
    # Action #
    select, pos = a['select'], a['pos']
    a1 = nn.Embed(5, ng)(select).reshape(B, N, 1, ng)  # (B, N, 1, Ng)
    a2 = nn.Embed(32*18+1, ng)(pos[...,0]*18+pos[...,1]).reshape(B, N, 1, ng)  # (B, N, 1, Ng)
    time_embd_g = nn.Embed(cfg.max_timestep+1, ng, embedding_init=nn.initializers.zeros)(timestep)  # (B, N) -> (B, N, Ng)
    # Reward #
    r = nn.tanh(Dense(ng)(jnp.expand_dims(r, -1))).reshape(B, N, 1, ng)  # (B, N) -> (B, N, Ng)
    # action1, action2, reward #
    xg = jnp.concatenate([a1, a2, r], 2).reshape(B, N*3, ng) + time_embd_g.repeat(3, 1)
    ### Embedding Local Token ###
    ### State ###
    cards = nn.Embed(cfg.n_cards, nl)(cards)  # (B, N, 5, Nl)
    elixir = nn.Embed(cfg.n_elixir+1, nl)(elixir).reshape(B, N, 1, nl)  # (B, N, 1, Nl)
    p1, p2 = self.cfg.patch_size
    arena = rearrange(arena, 'B N (H p1) (W p2) C -> B N (H W) (p1 p2 C)', p1=p1, p2=p2)
    P = H * W // p1 // p2
    img_embd = nn.Embed(P, nl, embedding_init=nn.initializers.zeros)(jnp.arange(P))  # (P, Nl)
    arena = Dense(nl)(arena) + img_embd  # (B, N, P, Nl)
    s = jnp.concatenate([arena, cards, elixir], -2)  # (B, N, 144+5+1, Nl)
    time_embd_l = nn.Embed(cfg.max_timestep+1, nl, embedding_init=nn.initializers.zeros)(timestep).reshape(B, N, 1, nl)  # (B, N) -> (B, N, Nl)
    xl = s + time_embd_l.repeat(s.shape[2], 2)  # (B, N, 150, Nl)
    ### ViDformer ###
    xl = nn.Dropout(cfg.p_drop_embd)(xl, deterministic=not train)
    xg = nn.Dropout(cfg.p_drop_embd)(xg, deterministic=not train)
    for i in range(cfg.n_block):
      xl, zg = ViDBlock(cfg=self.cfg)(xl, xg, train)
      zg = rearrange(zg, 'B (n N) Ng -> B Ng N n', n=4)
      if i != cfg.n_block - 1:  # xg.shape=(B, 3*N, Ng), get a1, a2, r
        xg = zg[...,:3]
      else:  # xg.shape=(B, N, Ng), get state
        xg = zg[...,3:4]
      xg = rearrange(xg, 'B Ng N n -> B (n N) Ng')
    xg = nn.LayerNorm()(xg)
    select = Dense(5, use_bias=False)(xg)
    pos = Dense(32*18, use_bias=False)(xg)
    return select, pos
    
  def get_state(self, train_cfg: TrainConfig, verbose: bool = False, train: bool = True) -> TrainState:
    def check_decay_params(kp, x):
      fg = x.ndim > 1
      for k in kp:
        if k.key in ['LayerNorm', 'Embed']:
          fg = False; break
      return fg
    def lr_fn():
      warmup_steps = train_cfg.warmup_tokens // (train_cfg.n_step * train_cfg.batch_size * train_cfg.accumulate)
      warmup_fn = optax.linear_schedule(0.0, train_cfg.lr, warmup_steps)
      second_steps = max(train_cfg.total_epochs * train_cfg.steps_per_epoch // train_cfg.accumulate - warmup_steps, 1)
      second_fn = optax.cosine_decay_schedule(
        train_cfg.lr, second_steps, 0.1
      )
      return optax.join_schedules(
        schedules=[warmup_fn, second_fn],
        boundaries=[warmup_steps]
      )
    drop_rng, verbose_rng, rng = jax.random.split(jax.random.PRNGKey(train_cfg.seed), 3)
    if not train:  # return state with apply function
      return TrainState.create(
        apply_fn=self.apply, params={'a': 1}, tx=optax.sgd(1),
        dropout_rng=rng, grads={}, accumulate=0, acc_count=0
      )
    # s, a, r, timestep
    B, l = train_cfg.batch_size, self.cfg.n_step
    s = {
      'arena': jnp.empty((B, l, 32, 18, 1+1+self.cfg.n_bar_size*2), int),
      'arena_mask': jnp.empty((B, l, 32, 18), bool),
      'cards': jnp.empty((B, l, 5,), int),
      'elixir': jnp.empty((B, l), int),
    }
    a = {
      'select': jnp.empty((B, l), int),
      'pos': jnp.empty((B, l, 2), int)
    }
    r, timestep = jnp.empty((B, l), float), jnp.empty((B, l), int)
    dummy = (s, a, r, timestep)
    if verbose: print(self.tabulate(verbose_rng, *dummy, train=False))
    variables = self.init(rng, *dummy, train=False)
    print("ViDformer params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variables)[0]]))
    decay_mask = jax.tree_util.tree_map_with_path(check_decay_params, variables['params'])
    train_cfg.lr_fn = lr_fn()
    state = TrainState.create(
      apply_fn=self.apply,
      params=variables['params'],
      # AdamW is Adam with weight decay
      tx=optax.chain(
        optax.clip_by_global_norm(train_cfg.clip_global_norm),
        optax.adamw(train_cfg.lr_fn, train_cfg.betas[0], train_cfg.betas[1], weight_decay=train_cfg.weight_decay, mask=decay_mask),
      ),
      dropout_rng=drop_rng,
      grads=variables['params'],
      accumulate=train_cfg.accumulate,
      acc_count=0,
    )
    return state
  
  def create_fns(self):
    def model_step(state: TrainState, s, a, r, timestep, y, train: bool):
      dropout_rng, base_rng = jax.random.split(state.dropout_rng)
      def loss_fn(params):
        # (B, l, 5), (B, l, 32*18)
        select, pos = state.apply_fn({'params': params}, s, a, r, timestep, train=train, rngs={'dropout': dropout_rng})
        y_select, y_pos = y['select'].reshape(-1), y['pos']
        y_pos = (y_pos[...,0] * 18 + y_pos[...,1]).reshape(-1)
        mask = y_select != 0
        n = mask.sum() + 1
        tmp = -jax.nn.log_softmax(select).reshape(-1, select.shape[-1])
        loss_select = tmp[jnp.arange(tmp.shape[0]), y_select].mean()
        tmp = -jax.nn.log_softmax(pos).reshape(-1, pos.shape[-1])
        loss_pos = (tmp[jnp.arange(tmp.shape[0]), y_pos] * mask).sum() / n
        B = r.shape[0]
        loss = B * (loss_select + loss_pos)

        flag_select = (jnp.argmax(select, -1).reshape(-1) == y_select)
        acc_select = flag_select.mean()
        acc_select_use = (flag_select * mask).sum() / n
        flag_pos = (jnp.argmax(pos, -1).reshape(-1) == y_pos)
        acc_pos = (flag_pos * mask).sum() /  n
        acc_select_and_pos = (flag_select * flag_pos * mask).sum() / n
        return loss, (loss_select, loss_pos, acc_select, acc_pos, acc_select_use, acc_select_and_pos)
      (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      state = accumulate_grads(state, grads)
      state = state.replace(dropout_rng=base_rng)
      return state, (loss, metrics)
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, s, a, r, timestep, step_len: Sequence[int] = None, rng: jax.Array = None, deterministic: bool = True):
      logits_select, logits_pos = state.apply_fn({'params': state.params}, s, a, r, timestep, train=False)
      if step_len is not None:
        logits_select = logits_select[jnp.arange(logits_select.shape[0]), step_len-1, :]  # (B, 5)
        logits_pos = logits_pos[jnp.arange(logits_pos.shape[0]), step_len-1, :]  # (B, 32*18)
      if deterministic:
        select = jnp.argmax(logits_select, -1, keepdims=True)
        pos = jnp.argmax(logits_pos, -1, keepdims=True)
      else:
        select = jax.random.categorical(rng, logits_select, -1)[..., None]
        pos = jax.random.categorical(rng, logits_pos, -1)[..., None]
      y, x = pos // 18, pos % 18
      return jnp.concatenate([select, x, y], -1), logits_select, logits_pos
    self.predict = jax.jit(predict, static_argnames='deterministic')

  def save_model(self, state, save_path):
    with open(save_path, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    print(f"Save weights to {save_path}")
  
if __name__ == '__main__':
  import os
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
  from katacr.constants.label_list import unit_list
  n_unit = len(unit_list)
  n_cards = 20
  n_step = 30
  max_timestep = 300
  gpt_cfg = ViDConfig(n_unit=n_unit, n_cards=n_cards, n_step=n_step, max_timestep=max_timestep)
  print(dict(gpt_cfg))
  gpt = ViDformer(gpt_cfg)
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.randint(rng, (batch_size, n_len), 0, 6)
  # print(gpt.tabulate(rng, x, train=False))
  # variable = gpt.init(rng, x, train=False)
  # print("params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variable)[0]]))
  train_cfg = TrainConfig(batch_size=1, steps_per_epoch=512, n_step=n_step, accumulate=64)
  state = gpt.get_state(train_cfg, verbose=True)
