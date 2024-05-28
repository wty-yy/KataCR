"""
Open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import os, sys
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from katacr.utils.ckpt_manager import CheckpointManager
from katacr.policy.offline.dataset import build_feature
import cv2, jax
from katacr.policy.env.interact_env import InteractEnv
from katacr.policy.env.video_env import VideoEnv
import numpy as np
from katacr.utils import colorstr, Stopwatch
from katacr.utils.ffmpeg.merge_videos import merge_videos_left_and_right
from katacr.constants.card_list import card2elixir
from katacr.policy.replay_data.data_display import GridDrawer
import time
from katacr.utils.csv_writer import CSVWriter

MODEL_NAME = "StARformer_3L_v0.7_golem_ai_cnn_blocks__nbc128__ep30__0__20240510_231638"
CSV_TITLE = ['episode', 'model_name', 'epoch', 'survival_time', 'total_reward', 'use_actions']
class_merge = [
  {'ice-spirit', 'ice-spirit-evolution'},
  {'skeleton', 'skeleton-evolution'}
]

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
      self, path_weights, load_epoch=None, vid_path=None, show=True, save=False,
      rtg=3., deterministic=True, verbose=False, show_predict=True, eval_num=None
    ):
    self.base_rtg, self.deterministic = rtg, deterministic
    self.verbose, self.show_predict, self.eval_num = verbose, show_predict, eval_num
    self.show, self.save = show, save
    if vid_path is not None:
      self.env = VideoEnv(vid_path, action_freq=2, show=show, verbose=verbose)
      self.path_save_dir = None
    else:
      self.env = InteractEnv(show=show, save=save)
      self.path_save_dir = self.env.path_save_dir
    self.rng = jax.random.PRNGKey(42)
    self.open_window = False
    self._load_model(path_weights, load_epoch)
    self.vid_writer = None
    self.csv_writer = CSVWriter(self.path_save_dir / f"{self.model_name}_load{self.load_epoch}.csv", title=CSV_TITLE)
  
  def _load_model(self, path_weights, load_epoch=None):
    print("Loading policy model...", end='')
    ckpt_mngr = CheckpointManager(str(path_weights))
    self.model_name = Path(path_weights).parent.name
    if load_epoch is not None:
      self.load_epoch = load_epoch
    else:
      self.load_epoch = int(sorted(Path(path_weights).glob('*'))[-1].name)
    load_info = ckpt_mngr.restore(self.load_epoch)
    params, cfg = load_info['variables']['params'], load_info['config']
    if 'StARformer' in str(path_weights):
      from katacr.policy.offline.starformer import StARformer, StARConfig, TrainConfig
      if 'cnn_mode' not in cfg:
        cfg['cnn_mode'] = 'resnet'
      self.model = StARformer(StARConfig(**cfg))
    if 'ViDformer' in str(path_weights):
      from katacr.policy.offline.vidformer import ViDformer, ViDConfig, TrainConfig
      self.model = ViDformer(ViDConfig(**cfg))
      self.model_name = 'ViDformer'
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.n_step = self.model.cfg.n_step
    self.n_bar_size = self.model.cfg.n_bar_size
    self._warmup()
    self.sw = [Stopwatch() for _ in range(2)]
    self.idx2card = {int(k): v for k , v in self.env.idx2card.items()}
    self.card2idx = {v: k for k, v in self.idx2card.items()}
    print("Complete!")
  
  def _init_sart(self):
    self.s = {k: [] for k in self.s_key}
    self.a = {k: [] for k in self.a_key}
    self.rtg = []
    self.timestep = []
  
  def _add_sart(self, s, a, rtg, timestep):
    ns, na = build_feature(s, a, use_card_idx=False)
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
  
  def get_action(self, cards):
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
    action, logits_select, logits_x, logits_y = jax.device_get(self.model.predict(
      self.state,
      {k: pad(v) for k, v in self.s.items()},
      {k: pad(v) for k, v in self.a.items()},
      pad(self.rtg),
      pad(self.timestep),
      step_len, rng, self.deterministic))
    prob_select_all = np.exp(logits_select-logits_select.max())[0]
    prob_select_all /= prob_select_all.sum()
    action = np.array(action[0])  # make a copy
    # Merge card and its evolution probability
    prob_select = []
    for i in range(len(cards)):
      prob = 0
      c = self.idx2card[cards[i]]
      cs = [c]
      for merge in class_merge:
        if c in merge: cs = merge
      prob = sum([prob_select_all[self.card2idx[c]] for c in cs])
      prob_select.append(prob)
    prob_select = np.array(prob_select)
    action[0] = np.argmax(prob_select) + 1
    for i in np.argsort(prob_select_all)[::-1]:
      print(f"{self.idx2card[i]}={prob_select_all[i]:.2f}", end=',')
    print()
    prob_select /= prob_select.sum()
    prob_x = np.exp(logits_x-logits_x.max())[0].reshape(1, 18)
    prob_x /= prob_x.sum()
    prob_y = np.exp(logits_y-logits_y.max())[0].reshape(32, 1)
    prob_y /= prob_y.sum()
    prob_pos = prob_y * prob_x
    prob_pos /= prob_pos.sum()
    # prob_pos = np.exp(logits_pos-logits_pos.max())[0].reshape(32, 18)
    # prob_pos /= prob_pos.sum()
    # if step_len == 30:
    #   np.save("/home/yy/Coding/GitHub/KataCR/logs/intercation/video1_eval_dataset_50.npy", data, allow_pickle=True)
    #   exit()
    prob_img = None
    if self.show_predict:
      sel_drawer = GridDrawer(1, 5, size=(576, 50))
      sel_drawer.paint((0, 0), (0,0,0), f"delay:\n{action[-1]:.0f}")  # future action
      for i in range(1,5):
        prob = prob_select[i-1]
        sel_drawer.paint((i, 0), (0, 0, int(255*prob)), f"{prob*100:.2f}")
      simg = np.array(sel_drawer.image)
      # if not self.open_window:
      #   cv2.namedWindow("Predict Select Probability", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      #   cv2.resizeWindow("Predict Select Probability", img.shape[:2][::-1])
      # cv2.imshow("Predict Select Probability", img)
      # cv2.waitKey(1)
      pos_drawer = GridDrawer(32, 18, size=(576, 846))
      for i in range(32):
        for j in range(18):
          prob = prob_pos[i,j]
          pos_drawer.paint((j, i), (0,0,int(255*prob)), f"{prob*100:.1f}")
      pimg = np.array(pos_drawer.image)
      prob_img = np.concatenate([pimg, simg], 0)
      # if not self.open_window:
      #   self.open_window = True
      #   cv2.namedWindow("Predict Probability", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      #   cv2.resizeWindow("Predict Probability", img.shape[:2][::-1])
      # cv2.imshow("Predict Probability", img)
      # cv2.waitKey(1)
      # if self.save:
      #   if self.vid_writer is None:
      #     self.path_save_vid = self.path_save_dir / f"{self.episode}_predict.mp4"
      #     self.vid_writer = cv2.VideoWriter(str(self.path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, (576, 896))
      #   self.vid_writer.write(img)
    return action, prob_img
  
  # def _init_vid_writer(self):
  #   if self.path_save_dir is None: return
  #   if self.vid_writer is not None:
  #     time.sleep(3)
  #     self.vid_writer.release()
  #     path_detection_vid = self.path_save_vid.with_stem(str(self.episode))
  #     path_origin_vid = self.path_save_vid.with_stem(str(self.episode)+'_org')
  #     path_merge_vid = merge_videos_left_and_right(path_detection_vid, self.path_save_vid)
  #     merge_videos_left_and_right(path_origin_vid, path_merge_vid)
  #     for p in [path_detection_vid, self.path_save_vid, path_origin_vid, path_merge_vid]:
  #       p.unlink()
  #     self.vid_writer = None
  
  def eval(self):
    self.episode = 0
    while True:
      scores, use_actions = [], 0
      self._init_sart()
      s, a, _ = self.env.reset(auto=self.eval_num is not None)
      self.episode += 1
      last_elixir = 0
      now_rtg, done = self.base_rtg, False
      self._add_sart(s, a, self.base_rtg, s['time'])
      while not done:
        if s['elixir'] is not None: last_elixir = s['elixir']
        with self.sw[0]:
          a, prob_img = self.get_action(s['cards'][1:])
        a = np.array(a)
        card = self.idx2card[s['cards'][a[0]]]
        if a[0] and card == 'empty':
          print(f"Skip action, since card index {a[0]} is 'empty'")
          a[0] = 0  # Skip
        if a[0] and card2elixir[card] > last_elixir:
          print(f"Skip action, since no enough elixir for card {card}={card2elixir[card]} > {last_elixir}")
          a[0] = 0  # Skip
        if last_elixir == 10:
          a[-1] = 0  # NO delay
        with self.sw[1]:
          s, _, r, done, info = self.env.step(a, max_delay=8, prob_img=prob_img)
          # s, a, r, done = self.env.step(a)
        if a[0]: use_actions += 1
        a = {'card_id': a[0], 'xy': a[1:3] if a[0] != 0 else None, 'delay': a[3]}
        # print("Action:", a)
        # When use future action predict, a[0] in [1,2,3,4]
        if self.verbose:
          print(colorstr("Time used (Eval):"), *[f"{k}={self.sw[i].dt*1e3:.1f}ms" for k, i in zip(['policy', 'step'], range(2))])
        now_rtg = max(now_rtg-r, 1)
        self._add_sart(s, a, now_rtg, s['time'])
        score = info['total_reward']
        scores.append(score)
      print(f"score {score}, timestep {s['time']}")
      import matplotlib.pyplot as plt
      plt.plot(scores)
      plt.tight_layout()
      plt.savefig(str(self.path_save_dir / f"scores_{self.episode}.png"))
      plt.close()
      self.csv_writer.write([self.episode, self.model_name, self.load_epoch, s['time'], score, use_actions])
      # self._init_vid_writer()
      if self.eval_num is not None and self.episode == self.eval_num:
        break
    print(colorstr("Finish all evaluation!"))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", default=MODEL_NAME,
    help="The policy model weights directory name in 'KataCR/logs/Policy/{model_name}'")
  parser.add_argument("--load-epoch", type=int, default=None,
    help="The load epoch id in 'KataCR/logs/Policy/{model_name}/ckpt/{load_epoch}'")
  parser.add_argument("--eval-num", type=int, default=20,
    help="The automatically evaluation number times")
  args = parser.parse_args()
  path_weights = path_root / f"logs/Policy/{args.model_name}/ckpt"
  evaluator = Evaluator(path_weights, load_epoch=args.load_epoch, show=True, save=True, deterministic=True, eval_num=args.eval_num)
  # vid_path = "/home/yy/Videos/CR_Videos/test/golem_ai/1.mp4"
  # evaluator = Evaluator(path_weights, vid_path, show=True, deterministic=False, verbose=False)
  evaluator.eval()

