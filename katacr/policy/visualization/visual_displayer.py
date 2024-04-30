"""
To open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import cv2
from katacr.policy.visualization.visual_fusion import VisualFusion
from katacr.utils import Stopwatch, second2str
from pathlib import Path
import numpy as np
import time

cap = cv2.VideoCapture(2)  # open stream
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
path_root = Path(__file__).parents[3]

class Displayer:
  def __init__(self, interval: int=1, save: bool=False):
    self.visual = VisualFusion()
    self.interval, self.save = interval, save
    if save:
      path_save_dir = path_root / "logs/visualize"
      path_save_dir.mkdir(exist_ok=True, parents=True)
      self.path_save = str(path_save_dir / time.strftime("%Y%m%d_%H%M%S.avi"))
      print(self.path_save)
  
  def __call__(self):
    while True:
      st = time.time()
      # for _ in range(self.interval):
      #   flag = cap.grab()
      flag, img = cap.read()
      if not flag: break
      rimg = self.visual.render(img, verbose=True)
      cv2.imshow("Detection", rimg)
      if self.save:
        if not hasattr(self, "writer"):
          self.writer = cv2.VideoWriter(self.path_save, cv2.VideoWriter_fourcc(*'MJPG'), 10, rimg.shape[:2][::-1])
        self.writer.write(rimg)
      # cv2.imshow("Detection", img)
      cv2.waitKey(1)
      print(f"Time used: {time.time() - st:4f}")

if __name__ == '__main__':
  visualizer = Displayer(save=True)
  visualizer()
