from katacr.policy.visual_fusion import VisualFusion
from katacr.policy.state_builder import StateBuilder
from katacr.policy.reward_builder import RewardBuilder
from katacr.yolov8.predict import ImageAndVideoLoader, Stopwatch, second2str
from pathlib import Path
import cv2, time
import numpy as np
import subprocess

BASE_DATA_INFO = dict(state=[], action=[], reward=[])

path_root = Path(__file__).parents[2]

class OfflineDatasetBuilder:
  def __init__(self, compress_threads=8):
    self.threads = compress_threads
    self.visual_fusion = VisualFusion()
    self.state_builder = StateBuilder()
    self.reward_builder = RewardBuilder()
    self.path_save_result = path_root / f"logs/offline" / time.strftime("%Y.%m.%d %H:%M:%S")
    self.path_save_result.mkdir(exist_ok=True, parents=True)
    self.reset()
  
  def reset(self, save_path=None):
    self.data = BASE_DATA_INFO.copy()
    if self.visual_fusion.yolo.tracker is not None:
      self.visual_fusion.yolo.tracker.reset()
    self.count = 0
    self.state_builder.reset()
    self.reward_builder.reset()
    if save_path is not None:
      self.save_data(save_path)
  
  def process(
      self, path, show=False, save=True, video_interval=3, save_freq=2,
      verbose=False):
    vid_writer, vid_path = None, None
    ds = ImageAndVideoLoader(path, video_interval=video_interval, cvt_part2=False)
    open_window = False
    sw = [Stopwatch() for _ in range(4)]
    for p, x, cap, s in ds:  # path, image ,capture, verbose string
      with sw[0]:
        visual_info = self.visual_fusion.process(x)
      if verbose:
        cv2.imwrite(str(self.path_save_result / f"debug_org.jpg"), x)
        cv2.imwrite(str(self.path_save_result / f"debug_det.jpg"), visual_info['arena'].show_box())
      with sw[1]:
        self.state_builder.update(visual_info)
      with sw[2]:
        self.reward_builder.update(visual_info)
      self.count += 1
      if self.count % save_freq == 0:
        with sw[3]:
          self.data['state'].append(self.state_builder.get_state(verbose=verbose))
          self.data['reward'].append(self.reward_builder.get_reward(verbose=verbose))
      else: sw[3].dt = 0
      # img = self.state_builder.render()
      if show or save:
        img = self.state_builder.render()
        r = self.data['reward'][-1] if len(self.data['reward']) else None
        img = self.reward_builder.render(img, r)
      save_path = str(self.path_save_result / (f"{Path(p).parent.name}_{Path(p).name}"))
      if ds.mode == 'image':
        if show:
          cv2.imshow('Detection', img)
          cv2.waitKey(0)
        if save:
          cv2.imwrite(save_path, img)
      else:  # video
        if vid_path != save_path:  # new video
          if vid_path is not None and self.visual_fusion.yolo.tracker is not None:
            self.reset(save_path)
          vid_path = save_path
          if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
          if cap:  # video
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = img.shape[:2]
          save_path = str(Path(save_path).with_suffix('.mp4'))
          vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps//video_interval, (w, h))
        if save:
          vid_writer.write(img)
      print(f"{s} {'+'.join([f'{s.dt * 1e3:.1f}' for s in sw])}ms", end='')
      
      if not open_window and show:
        open_window = True
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Detection', w, h)
      if show:
        cv2.imshow('Detection', img)
        cv2.waitKey(1)
      if ds.mode == 'video':
        second = (ds.total_frame - ds.frame) * sum([s.dt for s in sw])
        print(f", time left: {second2str(second)}", end='')
      print()
      # if self.count == 4:
      #   exit()
    if vid_path is not None:
      self.save_data(vid_path)
  
  def save_data(self, file_path):
    save_path = Path(file_path).with_suffix('')
    print("Building data...", end='')
    sw = Stopwatch()
    with sw:
      self.data['reward'] = np.array(self.data['reward'], np.float32)
      np.save(save_path, self.data, allow_pickle=True)
      save_path = str(save_path.with_suffix('.npy.xz'))
      subprocess.run(['xz', f'-T {self.threads}', save_path[:-3]])
    print("Time used:", sw.dt)
    print(f"Save data at {save_path}")

if __name__ == '__main__':
  odb = OfflineDatasetBuilder()
  # odb.process("/home/yy/Pictures/ClashRoyale/build_policy/multi_bar3.png")
  # odb.process("/home/yy/Videos/CR_Videos/test/test_feature_build2_sub.mp4", debug=True)
  # odb.process("/home/yy/Videos/CR_Videos/test/test_feature_build2.mp4", debug=True)
  # odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/lan77_20240406_episodes/2.mp4", debug=True)
  # odb.process("/home/yy/Videos/CR_Videos/test/lan77_20240406_ep_2_sub.mp4", debug=True)
  odb.process("/home/yy/Videos/CR_Videos/test/lan77_20240406_ep_2.mp4", verbose=True, show=True)
  # odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240410_132216_1_episodes/7.mp4", debug=True)
  # odb.process("/home/yy/Videos/CR_Videos/test/test_feature_build2_sub_end_sub.mp4", debug=True)
  # odb.process("/home/yy/Pictures/ClashRoyale/build_policy/multi_bar3.png", debug=True)
  # odb.process("/home/yy/Videos/CR_Videos/test/test_feature_build2_sub_sub.mp4", debug=True)
  # odb.state_builder.debug()