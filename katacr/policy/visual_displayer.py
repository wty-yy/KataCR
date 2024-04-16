"""
To open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import cv2
from katacr.policy.visual_fusion import VisualFusion
from katacr.utils import Stopwatch, second2str
from pathlib import Path
import numpy as np
import time

cap = cv2.VideoCapture(2)  # open stream
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

class Displayer:
  def __init__(self, interval: int=3):
    self.visual = VisualFusion()
    self.interval = interval
  
  def __call__(self):
    while True:
      for _ in range(self.interval):
        flag = cap.grab()
      flag, img = cap.retrieve()
      if not flag: break
      rimg = self.visual.render(img, verbose=True)
      cv2.imshow("Detection", rimg)
      # cv2.imshow("Detection", img)
      cv2.waitKey(1)

if __name__ == '__main__':
  visualizer = Displayer()
  visualizer()
