import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))
from katacr.policy.perceptron.sar_builder import SARBuilder
from katacr.yolov8.predict import ImageAndVideoLoader, Stopwatch, second2str
from pathlib import Path
import cv2, time
import numpy as np
import subprocess

path_root = Path(__file__).parents[3]

class OfflineDatasetBuilder:
  def __init__(self, compress_threads=8):
    self.threads = compress_threads
    self.sar_builder = SARBuilder()
    self.path_save_result = path_root / f"logs/offline" / time.strftime("%Y.%m.%d %H:%M:%S")
    self.path_save_result.mkdir(exist_ok=True, parents=True)
    self.reset()
  
  def reset(self, save_path=None):
    if save_path is not None:
      self.save_data(save_path)
    self.data = dict(state=[], action=[], reward=[])
    self.count = 0
    self.sar_builder.reset()
  
  def process(
      self, path, show=False, save=True, video_interval=3, save_freq=2,
      verbose=False):
    vid_writer, vid_path = None, None
    ds = ImageAndVideoLoader(path, video_interval=video_interval, cvt_part2=False)
    open_window = False
    dt = [0] * 5
    for p, x, cap, info in ds:  # path, image ,capture, verbose string
      results = self.sar_builder.update(x)
      if results is None: continue
      visual_info, dt[:4] = results
      if verbose:
        cv2.imwrite(str(self.path_save_result / f"debug_org.jpg"), x)
        cv2.imwrite(str(self.path_save_result / f"debug_det.jpg"), visual_info['arena'].show_box())
      self.count += 1
      if self.count % save_freq == 0:
        s, a, r, dt[4] = self.sar_builder.get_sar(verbose=verbose)
        for name, x in zip(('state', 'action', 'reward'), (s, a, r)):
          if name == 'action':
            offset = (x['offset'] - 1) // save_freq + 1  # ceil
            x = x.copy()
            x.pop('offset')
            if offset > 0:
              # assert self.data[name][-offset]['card_id'] == 0
              if self.data[name][-offset]['card_id'] != 0:
                print(f"ERROR: Action cover at frame={self.count} with offset={offset}")
              self.data[name][-offset] = x
              x = {'xy': None, 'card_id': 0}
          self.data[name].append(x)
      else: dt[4] = 0
      if show or save:
        img = self.sar_builder.render()
      save_path = str(self.path_save_result / (f"{Path(p).parent.name}_{Path(p).name}"))
      if ds.mode == 'image':
        if show:
          cv2.imshow('Detection', img)
          cv2.waitKey(0)
        if save:
          cv2.imwrite(str(Path(save_path).with_suffix('.jpg')), img)
      else:  # video
        if vid_path != save_path:  # new video
          if vid_path is not None:
            self.reset(vid_path)
          vid_path = save_path
          if save:
            if isinstance(vid_writer, cv2.VideoWriter):
              vid_writer.release()
            if cap:  # video
              fps = cap.get(cv2.CAP_PROP_FPS)
              h, w = img.shape[:2]
            save_path = str(Path(save_path).with_suffix('.mp4'))
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps//video_interval, (w, h))
        if save:
          vid_writer.write(img)
      print(f"{info} {'+'.join([f'{t * 1e3:.1f}' for t in dt])}ms", end='')
      
      if not open_window and show:
        open_window = True
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Detection', w, h)
      if show:
        cv2.imshow('Detection', img)
        cv2.waitKey(1)
      if ds.mode == 'video':
        second = (ds.total_frame - ds.frame) * sum(dt)
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
  # odb.process("/home/yy/Videos/CR_Videos/test/lan77_20240406_ep_2.mp4", verbose=True, show=False)
  # odb.process("/home/yy/Videos/CR_Videos/expert_videos/WTY_20240419_112947_1_golem_enermy_ai_episodes/5.mp4", verbose=True, show=False)
  # odb.process("/home/yy/Videos/CR_Videos/test/golem_ai/1.mp4", verbose=True, show=False)
  # odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240511_230101_golem_ai_episodes", verbose=True, show=False, save=False)
  # odb.process("/home/yy/Videos/CR_Videos/test/golem_ai/1_deploy_ice-golem.mp4", verbose=True, show=True, save=True)
  odb.process("/home/yy/Coding/GitHub/KataCR/logs/offline/list4.txt", verbose=True, show=False, save=False)
  # odb.process("/home/yy/Videos/CR_Videos/test/golem_ai/1.mp4", verbose=True, show=True, save=True)
  # odb.process("/home/yy/Videos/CR_Videos/test/golem_ai/1_two_action.mp4", verbose=True, show=True, save=True)
  # odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/lan77_20240406_episodes/1.mp4", verbose=True, show=True, save=True)
  # odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/lan77_20240406_episodes/*.mp4", verbose=True, show=False, save=False)
  # odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/lan77_20240406_episodes/4.mp4", verbose=True, show=False)
  #odb.process("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240410_132216_1_episodes/7.mp4", verbose=True, show=True)
  # odb.process("/home/yy/Pictures/ClashRoyale/build_policy/multi_bar3.png", debug=True)
  # odb.process("/home/yy/Videos/CR_Videos/test/test_feature_build2_sub_sub.mp4", debug=True)
  # odb.state_builder.debug()