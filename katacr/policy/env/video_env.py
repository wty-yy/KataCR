""" Test model on video data """
import cv2, time
import katacr.build_dataset.constant as const
from pathlib import Path
from katacr.utils import colorstr
from katacr.policy.perceptron.sar_builder import SARBuilder
from katacr.yolov8.predict import ImageAndVideoLoader, Stopwatch, second2str

class VideoEnv:
  def __init__(self, vid_path, video_interval=3, action_freq=2, show=True, save=False, verbose=True):
    self.show, self.save = show, save
    self.action_freq, self.verbose = action_freq, verbose
    self.dt = {'sar_update': None, 'sar_get': None, 'sar_total': None}
    self.done = True
    self.ds = ImageAndVideoLoader(vid_path, video_interval=video_interval, cvt_part2=False)
    self.sar_builder = SARBuilder()
    self.idx2card = self.sar_builder.visual_fusion.classifier.idx2card
    self.open_window = False
    self.reset()
  
  def _show_dt(self):
    if not self.verbose: return
    print(colorstr("Time used (Env):"), end='')
    tot = 0
    for k, t in self.dt.items():
      if t is None: continue
      print(f" {k}={t*1e3:.1f}ms", end='')
      tot += t
    print()
  
  def get_sar(self):
    for p, x, cap, info in self.ds:
      results = self.sar_builder.update(x)
      if results is None: continue
      dt = results[1]
      self.dt['sar_update'] = sum(dt)
      self.count += 1
      if self.show or self.save:
        rimg = self.sar_builder.render()
        rimg_size = rimg.shape[:2][::-1]
        if self.show:
          if not self.open_window:
            self.open_window = True
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Detection', rimg_size)
          cv2.imshow('Detection', rimg)
          cv2.waitKey(1)
      if self.count % self.action_freq == 0:
        s, a, r, dt = self.sar_builder.get_sar(verbose=False)
        self.dt['sar_get'] = dt
        self.dt['sar_total'] = self.dt['sar_get'] + self.dt['sar_update']
        done = self.ds.frame == self.ds.total_frame // 2 * 2
        if done and self.ds.total_frame % 2 == 1:
          next(self.ds)  # skip last frame
        # print(f"Frame: {self.ds.frame=} {self.ds.total_frame=}", info)
        if a['card_id']:
          print(colorstr('red', 'bold', f"Target Action (time={s['time']},frame={self.count}):"), a)
        return s, a, r, done
  
  def reset(self, auto=False):
    self.count = 0
    self.total_reward = 0
    self.sar_builder.reset()
    s, a, r, done = self.get_sar()
    self.total_reward += r
    self._show_dt()
    return s, a, r
  
  def step(self, action, max_delay=5, prob_img=None):
    """
    Args:
      action (np.ndarray): [select_index, position_xy, (delay)], shape=(3,) or (4,)
      max_delay (int): The max delay time.
    """
    assert self.done, "self.done=True, need reset first!"
    s, a, r, done = self.get_sar()
    self.total_reward += r
    if action[0] and (len(action) == 3 or (len(action) == 4 and action[-1] <= max_delay)):
      print(colorstr('yellow', 'bold', f"Predict Action (time={s['time']},frame={self.count},delay={action[-1]}):"), action)
    self._show_dt()
    return s, a, r, done, {'total_reward': self.total_reward}

if __name__ == '__main__':
  vid_path = "/home/yy/Videos/CR_Videos/test/golem_ai/3.mp4"
  env = VideoEnv(vid_path, save=False, verbose=False)
  env.reset()
  while True:
    # i = int(input())
    s, a, r, done = env.step([0, 9, 22])
    if done:
      env.reset()
