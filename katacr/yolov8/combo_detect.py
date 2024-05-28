from katacr.yolov8.train import YOLO_CR
from katacr.yolov8.predict import ImageAndVideoLoader
from pathlib import Path
from katacr.utils import Stopwatch, second2str
import time, cv2
from katacr.constants.label_list import unit2idx, idx2unit
import numpy as np
import torch, torchvision
from katacr.yolov8.custom_result import CRResults
from katacr.yolov8.custom_trackers import cr_on_predict_postprocess_end, cr_on_predict_start
from katacr.utils.detection.data import show_box

path_root = Path(__file__).parents[2]

path_detectors = [
  # '/home/yy/Coding/GitHub/KataCR/runs/detector1_v0.7.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector2_v0.7.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector1_v0.7.1.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector2_v0.7.1.pt',
  # '/home/yy/Coding/GitHub/KataCR/runs/detector3_v0.7.1.pt',
  # path_root / './runs/detector1_v0.7.6.pt',
  # path_root / './runs/detector2_v0.7.6.pt',
  # path_root / './runs/detector3_v0.7.6.pt',
  # path_root / './runs/detector1_v0.7.8.pt',
  # path_root / './runs/detector2_v0.7.8.pt',
  path_root / './runs/detector1_v0.7.13.pt',
  path_root / './runs/detector2_v0.7.13.pt',
]

class ComboDetector:
  def __init__(self, path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker='bytetrack'):
    self.models = [YOLO_CR(str(p)) for p in path_detectors]
    self.show_conf, self.conf = show_conf, conf
    self.iou_thre = iou_thre
    self.tracker = None
    if tracker == 'bytetrack':
      self.conf = 0.1
      self.tracker_cfg_path = str(path_root/'./katacr/yolov8/bytetrack.yaml')
      cr_on_predict_start(self, persist=True)
  
  def infer(self, x, pil=False):
    """
    Infer one image at once.

    Args:
      x (np.ndarray): Image with shape=(H,W,3).
      pil (bool): If taggled, input image is RGB mode else BGR.
    
    Returns:
      result (CRResult): with functions:
        - get_data(): get box information `xyxy, (track_id), conf, cls, bel`
        - show_box(verbose=False, show_conf=False, ...): return image with box (np.ndarray)
    """
    if pil: x = x[..., ::-1]  # RGB -> BGR
    results = [m.predict(x, verbose=False, conf=self.conf)[0] for m in self.models]
    preds = []
    for p in results:
      boxes = p.orig_boxes.clone()
      for i in range(len(boxes)):
        boxes[i, 5] = unit2idx[p.names[int(boxes[i, 5])]]
        preds.append(boxes[i])
    if not preds:
      preds = torch.zeros(0, 7)
    else:
      preds = torch.cat(preds, 0).reshape(-1, 7)
    i = torchvision.ops.nms(preds[:, :4], preds[:, 4], iou_threshold=self.iou_thre)
    preds = preds[i]
    # self.result will be used in `cr_on_predict_postprocess_end`
    self.result = CRResults(x, path="", names=idx2unit, boxes=preds)
    if self.tracker is not None:
      cr_on_predict_postprocess_end(self, persist=True)
    data = self.result.get_data()
    self.result.boxes.data = data[~(((data[:,0]>390) & (data[:,3]<120)) | ((data[:,2]<280) & (data[:,3]<80)))]
    return self.result

  def predict(self, source, show=False, save=True, video_interval=1):
    path_source = source
    vid_writer, vid_path = None, None
    ds = ImageAndVideoLoader(path_source, video_interval=video_interval)
    plot_kwargs = {'pil': True, 'font_size': None, 'conf': self.show_conf, 'line_width': None, 'font': str(path_root / 'utils/fonts/Arial.ttf')}
    open_window = False
    path_save_result = path_root / f"logs/detection" / time.strftime("%Y.%m.%d %H:%M:%S")
    if save:
      path_save_result.mkdir(exist_ok=True, parents=True)

    sw = Stopwatch()
    for p, x, cap, s in ds:  # path, image, capture, verbose string
      with sw:
        pred = self.infer(x)
      if ds.mode in ['image', 'video']:
        img = pred.show_box(verbose=False, show_conf=True)
        save_path = str(path_save_result / (f"{Path(p).parent.name}_ep{Path(p).name}"))
        if ds.mode == 'image':
          if show:
            cv2.imshow('Detection', img)
            cv2.waitKey(0)
          if save:
            cv2.imwrite(save_path, img)
        else:  # video
          if vid_path != save_path:  # new video
            if vid_path is not None and self.tracker is not None:
              self.tracker.reset()
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
        print(f"{s} {sw.dt * 1e3:.1f}ms", end='')
        
        if not open_window and show:
          open_window = True
          cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
          cv2.resizeWindow('Detection', w, h)
        if show:
          cv2.imshow('Detection', img)
          cv2.waitKey(1)
        if ds.mode == 'video':
          second = (ds.total_frame - ds.frame) * sw.dt
          print(f", time left: {second2str(second)}", end='')
        print()


if __name__ == '__main__':
  combo = ComboDetector(path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker='bytetrack')
  # combo = ComboDetector(path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker=None)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/OYASSU_20210528_episodes/1.mp4", show=True, save=True, video_interval=6)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/OYASSU_20230203_episodes/2.mp4", show=True, save=True, video_interval=3)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240218_episodes/1.mp4", show=False)
  # combo.predict("./logs/detection_files.txt", show=True, save=False)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240218_episodes/1.mp4", show=True, save=True, video_interval=3)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/segment_test/WTY_20240227_miners/1.mp4", show=True, save=True, video_interval=3)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/segment_test/WTY_20240222_8spells/1.mp4")
  # combo.predict("/home/yy/Coding/GitHub/KataCR/logs/split_video/OYASSU_20230203_episodes_2.mp4", show=True)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/segment_test/WTY_20240412/dagger0_cannoneer1_1.mp4", show=True, save=True, video_interval=3)
  # combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/lan77_20240406_episodes/2.mp4", show=True, save=False, video_interval=3)
  # combo.predict("/home/yy/Videos/CR_Videos/test/lan77_20240406_ep_2_sub_sub.mp4", show=True, save=True, video_interval=3)
  # combo.predict("/home/yy/Videos/CR_Videos/test/test_feature_build2_sub_end.mp4", show=True, save=True, video_interval=3)
  combo.predict("/home/yy/Coding/datasets/Clash-Royale-Dataset/videos/fast_pig_2.6/WTY_20240218_episodes/1.mp4", show=True, save=True, video_interval=3)
  # combo.predict("/home/yy/Videos/CR_Videos/musketeer_and_hogrider_insecond.mp4", show=True, save=True, video_interval=3)