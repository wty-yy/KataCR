from katacr.policy.feature_builder import VisualFusion, StateBuilder
from katacr.yolov8.predict import ImageAndVideoLoader, Stopwatch, second2str
from pathlib import Path
import cv2, time

path_root = Path(__file__).parents[2]

class OfflineDatasetBuilder:
  def __init__(self):
    self.visual_fusion = VisualFusion()
    self.state_builder = StateBuilder()
  
  def process(self, path, show=False, save=True, video_interval=3, save_freq=2, debug=False):
    vid_writer, vid_path = None, None
    ds = ImageAndVideoLoader(path, video_interval=video_interval, cvt_part2=False)
    open_window = False
    path_save_result = path_root / f"logs/offline" / time.strftime("%Y.%m.%d %H:%M:%S")
    if save or debug:
      path_save_result.mkdir(exist_ok=True, parents=True)
    sw = [Stopwatch(), Stopwatch()]
    for p, x, cap, s in ds:  # path, image ,capture, verbose string
      with sw[0]:
        visual_info = self.visual_fusion.process(x)
      if debug:
        cv2.imwrite(str(path_save_result / "debug_org.jpg"), x)
        cv2.imwrite(str(path_save_result / "debug_deet.jpg"), visual_info['arena'].show_box())
      with sw[1]:
        self.state_builder.update(visual_info)
      img = self.state_builder.debug()
      save_path = str(path_save_result / (f"{Path(p).parent.name}_ep{Path(p).name}"))
      if ds.mode == 'image':
        cv2.imshow('Detection', img)
        cv2.waitKey(0)
        if save:
          cv2.imwrite(save_path, img)
      else:  # video
        if vid_path != save_path:  # new video
          vid_path = save_path
          if self.visual_fusion.yolo.tracker is not None:
            self.visual_fusion.yolo.tracker.reset()
          if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
          if cap:  # video
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = img.shape[:2]
          save_path = str(Path(save_path).with_suffix('.mp4'))
          vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps//video_interval, (w, h))
        if save:
          vid_writer.write(img)
      print(f"{s} {sw[0].dt * 1e3:.1f}+{sw[1].dt * 1e3:.1f}ms", end='')
      
      if not open_window and show:
        open_window = True
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Detection', w, h)
      if show:
        cv2.imshow('Detection', img)
        cv2.waitKey(1)
      if ds.mode == 'video':
        second = (ds.total_frame - ds.frame) * (sw[0].dt + sw[1].dt)
        print(f", time left: {second2str(second)}", end='')
      print()

if __name__ == '__main__':
  odb = OfflineDatasetBuilder()
  # odb.process("/home/yy/Pictures/ClashRoyale/build_policy/multi_bar3.png")
  odb.process("/home/yy/Videos/CR_Videos/test/test_feature_build2_sub.mp4", debug=True)
  # odb.state_builder.debug()