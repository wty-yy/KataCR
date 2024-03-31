import multiprocessing, cv2

def image_show(queue: multiprocessing.Queue, width=568*2+50, height=896, name='Generation'):
  cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
  cv2.resizeWindow(name, int(width), int(height))
  while True:
    if not queue.empty():
      img = queue.get()
      cv2.imshow(name, img)
    cv2.waitKey(1)  # 1ms

def stream_show(queue: multiprocessing.Queue, stream_id=0, name='Stream', scale_ratio=0.5):
  open_window = False
  cap = cv2.VideoCapture(stream_id)
  while True:
    flag, img = cap.read()
    if not flag: break
    h, w, _ = img.shape
    while queue.qsize() > 1: queue.get()
    queue.put(img)
    if not open_window:
      open_window = False
      cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
      cv2.resizeWindow(name, int(w*scale_ratio), int(h*scale_ratio))
    cv2.imshow(name, img)
    cv2.waitKey(1)  # 1ms