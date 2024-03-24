import multiprocessing, cv2

def image_show(queue: multiprocessing.Queue, width=568*2+50, height=896, name='Generation'):
  cv2.namedWindow("Generation", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
  cv2.resizeWindow("Generation", int(width), int(height))
  while True:
    if not queue.empty():
      img = queue.get()
      cv2.imshow(name, img)
    cv2.waitKey(1)  # 10ms