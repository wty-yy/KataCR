import numpy as np
from pathlib import Path
import multiprocessing, cv2, time
from katacr.build_dataset.utils.split_part import process_part
from katacr.policy.env.utils import tap_screen
from katacr.utils.ffmpeg.format_conversion import compress_video
import shutil
DESTROY_FRAME_DELTA_THRE = 10  # wait 10*0.3s after game terminal

path_root = Path(__file__).parents[3]
path_save_dir = path_root / "logs/interaction" / time.strftime("%Y%m%d_%H:%M:%S")

class SARDaemon:
  WAIT_FOR_NEXT_EPISODE = 1
  AUTO_START_NEXT_EPISODE = 2
  def __init__(
      self,
      q_reset: multiprocessing.Queue,
      q_sar: multiprocessing.Queue,
      q_info: multiprocessing.Queue,
      q_prob_img: multiprocessing.Queue,
      show: bool = True, save: bool = False, interval: int = 2
    ):
    from katacr.policy.perceptron.sar_builder import SARBuilder
    self.q_reset, self.q_sar, self.q_info, self.q_prob_img = q_reset, q_sar, q_info, q_prob_img
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
    self.first_terminal_time = None
    self.total_reward = 0
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
    if self.first_terminal_time is None:
      ep_flag = self.ocr.process_center_texts(self.img)
      if ep_flag == self.ocr.END_EPISODE_FLAG:
        self.first_terminal_time = time.time()
        while not self.q_sar.empty(): self.q_sar.get()
    elif time.time() - self.first_terminal_time > DESTROY_FRAME_DELTA_THRE * 0.3:
        self.first_terminal_time = None
        self.terminal = True

  def _start_new_episode(self):
    # img_size = self.img.shape[:2][::-1]  # (1080, 2400)
    time.sleep(10)  # wait for OK button
    img_size = (1080, 2400)
    # End episode OK button
    tap_screen((534/1080,1940/2400), img_size=img_size, delay=10.0)
    # Tap empty space
    tap_screen((534/1080,65/2400), img_size=img_size, delay=0.5)
    # Menu button
    tap_screen((980/1080,320/2400), img_size=img_size, delay=0.5)
    # Training camp button
    tap_screen((648/1080,746/2400), img_size=img_size, delay=0.5)
    # Start episode OK button
    tap_screen((730/1080,1400/2400), img_size=img_size)

  def run(self):
    while True:
      ### Check If Reset ###
      reset = self.terminal
      while not self.q_reset.empty() or self.terminal:
        if self.vid_writer is not None:
          self.vid_writer.release()
          # self.vid_writer_org.release()
          self.vid_writer = None
          # self.vid_writer_org = None
          # print("RELEASE mp4 videos!!!")
          compress_video(self.path_save_vid)
        self.terminal = False
        self.total_reward = 0
        reset = True
        reset_id = self.q_reset.get()
        if reset_id == self.AUTO_START_NEXT_EPISODE:
          # print("Start new episode AUTO!")
          self._start_new_episode()
      if reset:
        self.episode += 1
        self.count = 0
        self.sar_builder.reset()
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
        self.total_reward += r
        info['dt']['sar_get'] = dt
        info['dt']['sar_total'] = info['dt']['sar_get'] + info['dt']['sar_update']
        info['total_reward'] = self.total_reward
        self._check_terminal()
        self.q_sar.put((s, a, r, self.terminal, info))
      ### Check If Show or Save ###
      if self.show or self.save:
        rimg = self.sar_builder.render()
        rimg_size = rimg.shape[:2][::-1]
        org_img_size = (int(rimg_size[1]/1280*600), rimg_size[1])
        org_img = cv2.resize(self.img, org_img_size)
        while not self.q_prob_img.empty():
          self.prob_img = self.q_prob_img.get()
        if not hasattr(self, 'prob_img'):
          self.prob_img = np.zeros((896, 576, 3), np.uint8)
        # print(rimg.shape, org_img.shape)
        rimg = np.concatenate([org_img, rimg, self.prob_img], 1)
        rimg_size = rimg.shape[:2][::-1]
        if self.show:
          if not self.open_window:
            self.open_window = True
            cv2.namedWindow('Agent', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Agent', rimg_size)
            # cv2.namedWindow('Origin', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv2.resizeWindow('Origin', org_img_size)
          cv2.imshow('Agent', rimg)
          # cv2.imshow('Origin', org_img)
          cv2.waitKey(1)
        if self.save:
          if self.vid_writer is None:
            self.path_save_vid = path_save_dir / f"{self.episode}.mp4"
            self.vid_writer = cv2.VideoWriter(str(self.path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, rimg_size)
            # path_save_vid = path_save_dir / f"{self.episode}_org.mp4"
            # self.vid_writer_org = cv2.VideoWriter(str(path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, org_img_size)
          self.vid_writer.write(rimg)
          # self.vid_writer_org.write(org_img)
