"""
Open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import cv2, time, subprocess, multiprocessing
import katacr.build_dataset.constant as const
from katacr.policy.perceptron.utils import cell2pixel, background_size
from pathlib import Path
from katacr.policy.env.sar_daemon import SARDaemon
from katacr.utils import colorstr
from katacr.policy.env.utils import tap_screen

def ratio2name(img_size):
  r = img_size[1] / img_size[0]
  for name, ratio in const.ratio.items():
    if ratio[0] <= r <= ratio[1]:
      return name
  raise Exception(f"Don't know ratio: {r:.4f}")

class InteractEnv:
  def __init__(self, show=True, save=False, verbose=False):
    """ Connect mobile phone """
    self.q_reset, self.q_sar, self.q_info, self.q_prob_img = [multiprocessing.Queue() for _ in range(4)]
    self.verbose = verbose
    multiprocessing.Process(target=SARDaemon, args=(self.q_reset, self.q_sar, self.q_info, self.q_prob_img, show, save), daemon=True).start()
    info = self.q_info.get()
    self.idx2card = info['idx2card']
    self.path_save_dir = info['path_save_dir']
    self.dt = {'sar_update': None, 'sar_get': None, 'sar_total': None}
    self.done = True
  
  def _show_dt(self):
    if not self.verbose: return
    print(colorstr("Time used (Env):"), end='')
    tot = 0
    for k, t in self.dt.items():
      if t is None: continue
      print(f" {k}={t*1e3:.1f}ms", end='')
      tot += t
    print()
  
  def reset(self, auto=False):
    self.q_reset.put(SARDaemon.AUTO_START_NEXT_EPISODE if auto else SARDaemon.WAIT_FOR_NEXT_EPISODE)
    s, a, r, done, info = self.q_sar.get()
    self.img_size = info['img_size']
    self.dt.update(info['dt'])
    assert not done
    self._show_dt()
    return s, a, r
  
  def _act(self, select, xy, delay=0.05):
    name = ratio2name(self.img_size)
    def get_xy(id, tap2part):
      part = f"part{id}_{name}"
      part2img = const.split_bbox_params[part]
      xy = [part2img[i]+part2img[i+2]*tap2part[i] for i in range(2)]
      return xy
    params = const.part3_bbox_params[select]
    tap2part = params[0]+params[2]/2, params[1]+params[3]/2
    tap_screen(get_xy(3, tap2part), img_size=self.img_size, delay=delay)
    params = cell2pixel(xy)
    tap2part = [params[i] / background_size[i] for i in range(2)]
    tap_screen(get_xy(2, tap2part), img_size=self.img_size)

  def step(self, action, max_delay=5, prob_img=None):
    """
    Args:
      action (np.ndarray): [select_index, position_xy, (delay)], shape=(3,) or (4,)
      max_delay (int): The max delay time.
      prob_img (bool): The probability image of the prediction action, send to SARDaemon
    """
    assert self.done, "self.done=True, need reset first!"
    act_time = None
    if prob_img is not None:
      self.q_prob_img.put(prob_img)
    if action[0]:
      if (max_delay is None) or (len(action) == 3 or (len(action) == 4 and action[-1] <= max_delay)):
        self._act(action[0], action[1:3])
        act_time = time.time()
    s, a, r, done, info = self.q_sar.get()
    while action[0] and not done and act_time is not None and info['timestamp'] < act_time:  # wait SAR after action complete
      s, a, r, done, info = self.q_sar.get()
    self.dt.update(info['dt'])
    self._show_dt()
    return s, a, r, done, info

if __name__ == '__main__':
  env = InteractEnv(save=True)
  env.reset()
  while True:
    i = int(input())
    s, a, r, done = env.step([i, 9, 22])
    if done:
      env.reset()
