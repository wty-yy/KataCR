"""
python generation.py
Your config format:
num_unit (int): Generation unit numbers.
intersection_ratio_thre (float): The max to keep intersection ratio.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from katacr.build_dataset.generator import Generator
import cv2
import numpy as np
from PIL import Image
import multiprocessing
from katacr.interact.utils import image_show

avail_names = {
  'small': ['king-tower', 'queen-tower', 'cannoneer-tower', 'tower-bar', 'king-tower-bar', 'bar', 'bar-level', 'clock', 'emote', 'text', 'elixir', 'selected', 'skeleton-king-bar', 'ice-spirit-evolution-symbol', 'evolution-symbol', 'bat', 'elixir-golem-small', 'fire-spirit', 'skeleton', 'lava-pup', 'skeleton-evolution', 'heal-spirit', 'ice-spirit', 'phoenix-egg', 'bat-evolution', 'minion', 'goblin', 'archer', 'spear-goblin', 'bomber', 'electro-spirit', 'royal-hog', 'rascal-girl', 'ice-spirit-evolution', 'hog', 'dirt', 'mini-pekka', 'wizard', 'zappy', 'barbarian', 'little-prince', 'firecracker', 'valkyrie', 'bandit', 'wall-breaker', 'musketeer', 'princess', 'guard', 'archer-evolution', 'goblin-brawler', 'bomber-evolution', 'elite-barbarian', 'bomb', 'goblin-ball', 'axe', 'electro-wizard', 'mother-witch', 'elixir-golem-mid', 'tesla'],
  'mid': ['king-tower', 'queen-tower', 'cannoneer-tower', 'tower-bar', 'king-tower-bar', 'bar', 'bar-level', 'clock', 'emote', 'text', 'elixir', 'selected', 'skeleton-king-bar', 'royal-recruit', 'ice-wizard', 'valkyrie-evolution', 'dart-goblin', 'mortar', 'firecracker-evolution', 'lumberjack', 'barbarian-evolution', 'royal-ghost', 'miner', 'night-witch', 'ram-rider', 'electro-dragon', 'hunter', 'the-log', 'mortar-evolution', 'executioner', 'mega-minion', 'witch', 'tesla-evolution', 'golemite', 'barbarian-barrel', 'cannon-cart', 'monk', 'skeleton-dragon', 'magic-archer', 'ice-golem', 'knight-evolution', 'royal-recruit-evolution', 'knight', 'hog-rider', 'battle-ram', 'baby-dragon', 'fisherman', 'goblin-drill', 'rascal-boy', 'cannon', 'prince', 'lava-hound', 'tombstone', 'wall-breaker-evolution', 'dark-prince', 'elixir-collector', 'archer-queen', 'battle-healer', 'goblin-barrel', 'tesla-evolution-shock'],
  'big': ['king-tower', 'queen-tower', 'cannoneer-tower', 'tower-bar', 'king-tower-bar', 'bar', 'bar-level', 'clock', 'emote', 'text', 'elixir', 'selected', 'skeleton-king-bar', 'phoenix-small', 'fireball', 'golden-knight', 'bowler', 'goblin-cage', 'pekka', 'x-bow', 'inferno-dragon', 'sparky', 'rocket', 'mega-knight', 'flying-machine', 'skeleton-barrel', 'phoenix-big', 'elixir-golem-big', 'royal-guardian', 'barbarian-hut', 'golem', 'goblin-giant', 'goblin-hut', 'mighty-miner', 'balloon', 'giant', 'furnace', 'skeleton-king', 'inferno-tower', 'bomb-tower', 'royal-giant', 'royal-delivery', 'giant-snowball', 'royal-giant-evolution', 'giant-skeleton', 'electro-giant', 'zap', 'rage', 'clone', 'freeze', 'lightning', 'poison', 'earthquake', 'skeleton-king-skill', 'graveyard', 'arrows', 'tornado'],
}

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
            if n not in ['avail=']:
              setattr(self, w, eval(k.split('=')[-1]))
            else:
              setattr(self, w, avail_names[k.split('=')[-1]])
      # self.generator.background_index = self.background
      # self.generator.intersect_ratio_thre = self.intersection_ratio_thre
      # self.generator.reset()
      if isinstance(self.avail_names, str): self.avail_names = avail_names[self.avail_names]
      self.generator = Generator(background_index=self.background, intersect_ratio_thre=self.intersection_ratio_thre, avail_names=self.avail_names)
      print(self.num_unit, self.intersection_ratio_thre, self.background, self.avail_names)

if __name__ == '__main__':
  displayer = Displayer()
  displayer()

"""


"""