"""
open video:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
from paddleocr import PaddleOCR, draw_ocr
from katacr.interact.utils import stream_show
import time, cv2, multiprocessing, subprocess
from pathlib import Path
from katacr.utils import Stopwatch
path_root = Path(__file__).parents[2]

def call_shell(cmd):
  output = subprocess.check_output(cmd, shell=True)
  print("CMD", cmd)
  print("Output", output)
  return output

class OCRDisplayer:
  def __init__(self, stream_id=2):
    self.stream_id =  stream_id
    self.ocr = PaddleOCR(use_angle_cls=False, lang='en')
    self.screen_queue = multiprocessing.Queue()
    self.screen_process = multiprocessing.Process(target=stream_show, args=(self.screen_queue, self.stream_id))
    self.screen_process.daemon = True
    self.screen_process.start()
    self.sw = Stopwatch()

  def __call__(self):
    open_window = False
    while True:
      start_time = time.time()
      if not self.screen_queue.empty():
        img = self.screen_queue.get()
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2))
        cv2.imshow("Resize", img)
        cv2.waitKey(1)
        # examp: [[[521.0, 1208.0], [557.0, 1208.0], [557.0, 1245.0], [521.0, 1245.0]], ('良', 0.9963672161102295)]
        with self.sw:
          result = self.ocr.ocr(img)[0]
        print("OCR Time used:", self.sw.dt)
        if result is not None and len(result):
          boxes = [line[0] for line in result]
          print(boxes)
          txts = [line[1][0] for line in result]
          print(txts)
          scores = [line[1][1] for line in result]
          img_show = draw_ocr(img, boxes, txts, scores, font_path=str(path_root / "katacr/utils/fonts/SimHei.ttf"))
          h, w, _ = img_show.shape
          # print(img.shape)
          print("Time used:", time.time() - start_time)
          start_time = time.time()
          if not open_window:
            open_window = True
            cv2.namedWindow("OCR", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OCR", int(w), int(h))
          cv2.imshow("OCR", img_show)
          cv2.waitKey(1)

if __name__ == '__main__':
  ocr_displayer = OCRDisplayer()
  ocr_displayer()
