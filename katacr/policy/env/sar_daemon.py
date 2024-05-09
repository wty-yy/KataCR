import multiprocessing, cv2, time
from pathlib import Path
from katacr.build_dataset.utils.split_part import process_part

path_root = Path(__file__).parents[3]
path_save_dir = path_root / "logs/intercation" / time.strftime("%Y%m%d_%H:%M:%S")

class SARDaemon:
  def __init__(
      self,
      q_reset: multiprocessing.Queue,
      q_sar: multiprocessing.Queue,
      q_info: multiprocessing.Queue,
      show: bool = True, save: bool = False, interval: int = 2
    ):
    from katacr.policy.perceptron.sar_builder import SARBuilder
    self.q_reset, self.q_sar, self.q_info = q_reset, q_sar, q_info
    self.show, self.save = show, save
    self.interval = interval
    self.sar_builder = SARBuilder()
    self.q_info.put({'idx2card': self.sar_builder.visual_fusion.classifier.idx2card, 'path_save_dir': path_save_dir})
    self.ocr = self.sar_builder.action_builder.ocr  # Share OCR with action text OCR
    self.cap = cv2.VideoCapture(2)
    assert self.cap.isOpened(), "The phone stream can't connect!"
    self.open_window = False
    self.vid_writer = None
    if self.save:
      path_save_dir.mkdir(exist_ok=True, parents=True)
    self.episode = 0
    self.terminal = True
    self.run()
  
  def _read_img(self):
    self.timestamp = time.time()
    _, img = self.cap.read()
    return img

  def _wait_next_episode(self):
    """ Wait for entering one episode """
    start_time = time.time()
    while self.cap.isOpened():
      img = self._read_img()
      t = self.ocr.process_part1(process_part(img, 1))
      ep_flag = self.ocr.process_center_texts(img)
      if ep_flag == self.ocr.START_EPISODE_FLAG and t < 10:
        return img
      time.sleep(0.05)
      if time.time() - start_time > 2:
        print("Wait for opening new episode...")
        start_time = time.time()

  def _check_terminal(self):
    if not self.terminal:
      ep_flag = self.ocr.process_center_texts(self.img)
      if ep_flag == self.ocr.END_EPISODE_FLAG:
        self.terminal = True
        while not self.q_sar.empty(): self.q_sar.get()

  def run(self):
    while True:
      ### Check If Reset ###
      reset = self.terminal
      while not self.q_reset.empty() or self.terminal:
        if self.vid_writer is not None:
          self.vid_writer.release()
          self.vid_writer_org.release()
          self.vid_writer = None
        self.terminal = False
        reset = True
        self.q_reset.get()
      if reset:
        self.episode += 1
        self.count = 0
        self.sar_builder.reset()
        self.vid_writer = None
        self.vid_writer_org = None
        self.img = self._wait_next_episode()
      else:
        self.img = self._read_img()
      ### Get SAR and Terminal ###
      while self.q_sar.qsize() > 1: self.q_sar.get()
      results = self.sar_builder.update(self.img)
      if results is None: continue
      dt = results[1]
      self.count += 1
      if self.count % self.interval == 0:
        info = {'dt': {}, 'timestamp': self.timestamp, 'img_size': self.img.shape[:2][::-1]}
        info['dt']['sar_update'] = sum(dt)
        s, a, r, dt = self.sar_builder.get_sar(verbose=False)
        info['dt']['sar_get'] = dt
        info['dt']['sar_total'] = info['dt']['sar_get'] + info['dt']['sar_update']
        self._check_terminal()
        self.q_sar.put((s, a, r, self.terminal, info))
      ### Check If Show or Save ###
      if self.show or self.save:
        rimg = self.sar_builder.render()
        rimg_size = rimg.shape[:2][::-1]
        if self.show:
          if not self.open_window:
            self.open_window = True
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Detection', rimg_size)
          cv2.imshow('Detection', rimg)
          cv2.waitKey(1)
        if self.save:
          if self.vid_writer is None:
            path_save_vid = path_save_dir / f"{self.episode}.mp4"
            self.vid_writer = cv2.VideoWriter(str(path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, rimg_size)
            path_save_vid = path_save_dir / f"{self.episode}_org.mp4"
            self.vid_writer_org = cv2.VideoWriter(str(path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, (600, 1280))
          self.vid_writer.write(rimg)
          self.vid_writer_org.write(cv2.resize(self.img, (600, 1280)))
