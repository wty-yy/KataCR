"""
python generation.py
Your config format:
num_unit (int): Generation unit numbers.
intersection_ratio_thre (float): The max to keep intersection ratio.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from katacr.build_dataset.generator import Generator
import cv2
import numpy as np
from PIL import Image
import multiprocessing
from katacr.interact.utils import image_show

class Displayer:
  def __init__(self):
    self.num_unit = 40
    self.intersection_ratio_thre = 0.5
    self.background = None
    self.avail_names = None
    self.generator = Generator()
    self.show_queue = multiprocessing.Queue()
    self.process = multiprocessing.Process(target=image_show, args=(self.show_queue,))
    self.process.start()
  
  def __call__(self):
    while True:
      self.generator.add_tower()
      self.generator.add_unit(self.num_unit)
      _, _, img2 = self.generator.build(show_box=True)
      _, _, img1 = self.generator.build(show_box=False)

      img1, img2 = np.array(img1), np.array(img2)
      img = np.concatenate([img1, np.zeros((img1.shape[0], 10, img1.shape[2])), img2], 1).astype(np.uint8)
      # Image.fromarray(img).show()
      self.show_queue.put(img[...,::-1])
      s = input("Your config (nu=, ithre=, back=, avail=): ")
      for k in s.split(' '):
        for n, w in zip(['nu=', 'ithre=', 'back=', 'avail='], ['num_unit', 'intersection_ratio_thre', 'background', 'avail_names']):
          if n in k:
            setattr(self, w, eval(k.split('=')[-1]))
      self.generator.background_index = self.background
      self.generator.intersect_ratio_thre = self.intersection_ratio_thre
      self.generator.reset()
      self.generator = Generator(background_index=self.background, intersect_ratio_thre=self.intersection_ratio_thre, avail_names=self.avail_names)
      print(self.num_unit, self.intersection_ratio_thre, self.background, self.avail_names)

if __name__ == '__main__':
  displayer = Displayer()
  displayer()

"""
small: ['king-tower', 'queen-tower', 'cannoneer-tower', 'tower-bar', 'king-tower-bar', 'bar', 'bar-level', 'clock', 'emote', 'text', 'elixir', 'selected', 'skeleton-king-bar', 'ice-spirit-evolution-symbol', 'evolution-symbol', 'bat', 'elixir-golem-small', 'fire-spirit', 'skeleton', 'lava-pup', 'skeleton-evolution', 'heal-spirit', 'ice-spirit', 'phoenix-egg', 'bat-evolution', 'minion', 'goblin', 'archer', 'spear-goblin', 'bomber', 'electro-spirit', 'royal-hog', 'rascal-girl', 'ice-spirit-evolution', 'hog', 'dirt', 'mini-pekka', 'wizard', 'zappy', 'barbarian', 'little-prince', 'firecracker', 'valkyrie', 'bandit', 'wall-breaker', 'musketeer', 'princess', 'guard', 'archer-evolution', 'goblin-brawler', 'bomber-evolution', 'elite-barbarian', 'bomb', 'goblin-ball', 'axe', 'electro-wizard', 'mother-witch', 'elixir-golem-mid', 'tesla']

"""