"""
Open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
from katacr.policy.offline.starformer import StARformer, StARConfig, TrainConfig
from katacr.utils.ckpt_manager import CheckpointManager
from katacr.policy.offline.dataset import build_feature
import cv2, jax
from katacr.policy.env.interact_env import InteractEnv
from katacr.policy.env.video_env import VideoEnv
from pathlib import Path
import numpy as np
from katacr.utils import colorstr, Stopwatch
from katacr.constants.card_list import card2elixir
from katacr.policy.replay_data.data_display import GridDrawer

path_root = Path(__file__).parents[3]
path_weights = path_root / "logs/Policy/StARformer-WeightSample__0__20240504_225245/ckpt"
# path_weights = path_root / "logs/Policy/StARformer-test-data1__0__20240504_170431/ckpt"

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
  """ This function would pad at the end of certain axis, https://stackoverflow.com/a/49766444 """
  pad_size = target_length - array.shape[axis]
  if pad_size <= 0:
    return array
  npad = [(0, 0)] * array.ndim
  npad[axis] = (0, pad_size)
  return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

class Evaluator:
  s_key = ['arena', 'arena_mask', 'cards', 'elixir']
  a_key = ['select', 'pos']

  def __init__(
      self, path_weights=path_weights, vid_path=None, show=True, save=False,
      rtg=3, deterministic=True, verbose=False, show_predict=True
    ):
    self.base_rtg, self.deterministic = rtg, deterministic
    self.verbose, self.show_predict = verbose, show_predict
    if vid_path is not None:
      self.env = VideoEnv(vid_path, action_freq=3, show=show, verbose=verbose)
    else:
      self.env = InteractEnv(show=show, save=save)
    self.rng = jax.random.PRNGKey(42)
    self.open_window = False
    self._load_model(path_weights)
  
  def _load_model(self, path_weights):
    print("Loading policy model...", end='')
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
    self.idx2card = self.env.idx2card
    print("Complete!")
  
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
    step_len = min(len(self.timestep), n_step)  # FIX BUG: Don't use len(self.s)
    rng, self.rng = jax.random.split(self.rng)
    # for k, v in self.s.items():
    #   print(f"{k}: {pad(v).shape}")
    # for k, v in self.a.items():
    #   print(f"{k}: {pad(v).shape}")
    # print(pad(self.rtg).shape, pad(self.timestep).shape, step_len, rng, self.deterministic)
    data = {
      's': {k: pad(v) for k, v in self.s.items()},
      'a': {k: pad(v) for k, v in self.a.items()},
      'rtg': pad(self.rtg),
      'timestep': pad(self.timestep),
    }
    action, logits_pos = jax.device_get(self.model.predict(
      self.state,
      {k: pad(v) for k, v in self.s.items()},
      {k: pad(v) for k, v in self.a.items()},
      pad(self.rtg),
      pad(self.timestep),
      step_len, rng, self.deterministic))
    action = action[0]
    prob_pos = np.exp(logits_pos)[0].reshape(32, 18)
    prob_pos /= prob_pos.sum()
    # if step_len == 30:
    #   np.save("/home/yy/Coding/GitHub/KataCR/logs/intercation/video1_eval_dataset_50.npy", data, allow_pickle=True)
    #   exit()
    if self.show_predict:
      pos_drawer = GridDrawer()
      for i in range(32):
        for j in range(18):
          prob = prob_pos[i,j]
          pos_drawer.paint((j, i), (0,0,int(255*prob)), f"{prob*100:.2f}")
      img = np.array(pos_drawer.image)
      if not self.open_window:
        self.open_window = True
        cv2.namedWindow("Predict Position Probability", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Predict Position Probability", img.shape[:2][::-1])
      cv2.imshow("Predict Position Probability", img)
      cv2.waitKey(1)
    return action
  
  def eval(self):
    score = 0
    while True:
      self._init_sart()
      s, a, _ = self.env.reset()
      last_elixir = 0
      now_rtg, done = self.base_rtg, False
      self._add_sart(s, a, self.base_rtg, s['time'])
      while not done:
        if s['elixir'] is not None: last_elixir = s['elixir']
        with self.sw[0]:
          a = self.get_action()
        a = np.array(a)
        # card = self.idx2card[str(s['cards'][a[0]])]
        # if a[0] and card == 'empty':
        #   print(f"Skip action, since card index {a[0]} is 'empty'")
        #   a[0] = 0  # Skip
        # if a[0] and card2elixir[card] > last_elixir:
        #   print(f"Skip action, since no enough elixir for card {card}={card2elixir[card]} > {last_elixir}")
        #   a[0] = 0  # Skip
        with self.sw[1]:
          # s, _, r, done = self.env.step(a)
          s, a, r, done = self.env.step(a)
        # a = {'card_id': a[0], 'xy': a[1:3] if a[0] != 0 else None}
        if self.verbose:
          print(colorstr("Time used (Eval):"), *[f"{k}={self.sw[i].dt*1e3:.1f}ms" for k, i in zip(['policy', 'step'], range(2))])
        now_rtg = max(now_rtg-r, 1)
        self._add_sart(s, a, now_rtg, s['time'])
        score += r
      print(f"score {score}, timestep {s['time']}")

if __name__ == '__main__':
  # evaluator = Evaluator(path_weights, show=True, save=True, deterministic=True)
  vid_path = "/home/yy/Videos/CR_Videos/test/golem_ai/1.mp4"
  evaluator = Evaluator(path_weights, vid_path, show=True, deterministic=True, verbose=False)
  evaluator.eval()

