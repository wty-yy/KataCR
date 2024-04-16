"""
To open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import cv2
from katacr.yolov8.combo_detect import ComboDetector
from katacr.utils import Stopwatch, second2str
from pathlib import Path
import numpy as np
import time

path_root = Path(__file__).parents[2]
path_detectors = [
  path_root / './runs/detector1_v0.7.10.pt',
  path_root / './runs/detector2_v0.7.10.pt',
]
cap = cv2.VideoCapture(2)  # open stream
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

class Visualizer:
  def __init__(self, interval: int=3):
    self.model = ComboDetector(path_detectors)
    self.interval = interval
  
  def __call__(self):
    while True:
      for _ in range(self.interval):
        flag = cap.grab()
      flag, img = cap.retrieve()
      if not flag: break
      cv2.imshow("Detection", img)

if __name__ == '__main__':
  visualizer = Visualizer()
  visualizer()
