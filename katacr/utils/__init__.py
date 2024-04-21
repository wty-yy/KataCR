from PIL import Image
import numpy as np
import contextlib, time
import sys
from pathlib import Path

def load_image_array(
    path_image, to_gray=False,
    keep_dim=True, resize=None
  ):
  image = Image.open(path_image)
  if resize is not None: image = image.resize(resize)
  if to_gray: image = image.convert("L")
  image = np.array(image)
  if keep_dim and image.ndim == 2: image = image[..., None]
  return image 

"""
# A stopwath is used to metric code block running time.
# Usage:
sw = Stopwatch()
with sw:
  your code...
  ...
print("delta time:", sw.dt, "total time:", sw.t)
"""
class Stopwatch(contextlib.ContextDecorator):
  def __init__(self, t=0.0):
    self.t = t
    self.avg_dt = 0
    self.avg_per_s = 0
    self.count = 0
    self.dt = 0
  
  def __enter__(self):
    self.start = time.time()
    return self
  
  def __exit__(self, *args):
    self.dt = time.time() - self.start
    self.t += self.dt
    self.count += 1
    self.avg_dt += (self.dt - self.avg_dt) / self.count
    self.avg_per_s += (1 / self.dt - self.avg_per_s) / self.count

def colorstr(*input):
  # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
  *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
  colors = {
    'black': '\033[30m',  # basic colors
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bright_black': '\033[90m',  # bright colors
    'bright_red': '\033[91m',
    'bright_green': '\033[92m',
    'bright_yellow': '\033[93m',
    'bright_blue': '\033[94m',
    'bright_magenta': '\033[95m',
    'bright_cyan': '\033[96m',
    'bright_white': '\033[97m',
    'end': '\033[0m',  # misc
    'bold': '\033[1m',
    'underline': '\033[4m'}
  return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

# How to redirect stdout to both file and console with scripting?
# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger:
  def __init__(self, path: str):
    Path(path).parent.mkdir(exist_ok=True)
    self.terminal = sys.stdout
    self.log = open(path, "a")
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  
  def flush(self):
    pass    
# usage: sys.stdout = Logger()

# Convert second to HOUR:MINUTE:SECOND  # if HOUR==0: Don't display this
def second2str(second):
  s = int(second)
  m = int(second // 60)
  h = int(m // 60)
  ret = ""
  if h:
    ret += f"{h:02}:"
    m = int(m % 60)
  s = int(s % 60)
  ret += f"{m:02}:{s:02}"
  return ret

class Config:
  def __iter__(self):
    for name in dir(self):
      val = getattr(self, name)
      if '__' not in name:
        yield (name, val)

  def __repr__(self):
    return str(dict(self))

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    