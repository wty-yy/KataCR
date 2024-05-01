import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
from katacr.policy.offline.starformer import StARformer, StARConfig, TrainConfig
from katacr.utils.ckpt_manager import CheckpointManager
from katacr.policy.offline.dataset import build_feature
import cv2, jax
from katacr.policy.interactor.env import InteractEnv
from pathlib import Path
import numpy as np
from katacr.utils import colorstr, Stopwatch

path_root = Path(__file__).parents[3]
path_weights = path_root / "logs/Policy/StARformer__0__20240501_125642/ckpt"

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
  """ This function would pad at the end of certain axis, https://stackoverflow.com/a/49766444 """
  pad_size = target_length - array.shape[axis]
  if pad_size <= 0:
    return array
  npad = [(0, 0)] * array.ndim
  npad[axis] = (0, pad_size)
  return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

class Evaulator:
  s_key = ['arena', 'arena_mask', 'cards', 'elixir']
  a_key = ['select', 'pos']

  def __init__(self, path_weights, show=True, save=False, rtg=3, deterministic=True):
    self.base_rtg, self.deterministic = rtg, deterministic
    self.env = InteractEnv(show=show, save=save)
    self.rng = jax.random.PRNGKey(42)
    self._load_model(path_weights)
  
  def _load_model(self, path_weights):
    ckpt_mngr = CheckpointManager(str(path_weights))
    load_step = int(sorted(Path(path_weights).glob('*'))[-1].name)
    load_info = ckpt_mngr.restore(load_step)
    params, cfg = load_info['variables']['params'], load_info['config']
    self.model = StARformer(StARConfig(**cfg))
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.n_step = self.model.cfg.n_step
    self.n_bar_size = self.model.cfg.n_bar_size
    self._warmup()
    self.sw = [Stopwatch() for _ in range(2)]
  
  def _init_sart(self):
    self.s = {k: [] for k in self.s_key}
    self.a = {k: [] for k in self.a_key}
    self.rtg = []
    self.timestep = []
  
  def _add_sart(self, s, a, rtg, timestep):
    ns, na = build_feature(s, a)
    for x, nx in zip([self.s, self.a], [ns, na]):
      for k in x.keys():
        x[k].append(nx[k])
    self.rtg.append(np.array(rtg, np.float32))
    timestep = min(timestep, self.model.cfg.max_timestep)
    self.timestep.append(np.array(timestep, np.int32))
  
  def _warmup(self):
    B, l = 1, self.n_step
    s = {
      'arena': np.empty((B, l, 32, 18, 1+1+self.n_bar_size*2), np.int32),
      'arena_mask': np.empty((B, l, 32, 18), np.bool_),
      'cards': np.empty((B, l, 5,), np.int32),
      'elixir': np.empty((B, l), np.int32),
    }
    a = {
      'select': np.empty((B, l), np.int32),
      'pos': np.empty((B, l, 2), np.int32)
    }
    r, timestep = np.empty((B, l), np.float32), np.empty((B, l), np.int32)
    self.model.predict(
      self.state,
      s, a, r, timestep, l, self.rng, self.deterministic)
  
  def get_action(self):
    n_step = self.n_step
    def pad(x):
      x = np.expand_dims(np.stack(x[-n_step:]), 0)
      return pad_along_axis(x, n_step, 1)
    step_len = min(len(self.s), n_step)
    rng, self.rng = jax.random.split(self.rng)
    # for k, v in self.s.items():
    #   print(f"{k}: {pad(v).shape}")
    # for k, v in self.a.items():
    #   print(f"{k}: {pad(v).shape}")
    # print(pad(self.rtg).shape, pad(self.timestep).shape, step_len, rng, self.deterministic)
    action = jax.device_get(self.model.predict(
      self.state,
      {k: pad(v) for k, v in self.s.items()},
      {k: pad(v) for k, v in self.a.items()},
      pad(self.rtg),
      pad(self.timestep),
      step_len, rng, self.deterministic))[0]
    return action
  
  def eval(self):
    score = 0
    while True:
      self._init_sart()
      s, a, _ = self.env.reset()
      now_rtg, done = self.base_rtg, False
      self._add_sart(s, a, self.base_rtg, s['time'])
      while not done:
        with self.sw[0]:
          a = self.get_action()
        with self.sw[1]:
          s, a, r, done = self.env.step(a)
        print(colorstr("Time used (Eval):"), *[f"{k}={self.sw[i].dt*1e3:.1f}ms" for k, i in zip(['policy', 'step'], range(2))])
        now_rtg -= r
        self._add_sart(s, a, now_rtg, s['time'])
        score += r
      print(f"score {score}, timestep {s['time']}")

if __name__ == '__main__':
  evaluator = Evaulator(path_weights, show=True, save=True)
  evaluator.eval()

