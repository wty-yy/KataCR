from PIL import Image
import numpy as np
import contextlib, time

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

class Stopwatch(contextlib.ContextDecorator):
  def __init__(self, t=0.0):
    self.t = t
  
  def __enter__(self):
    self.start = time.time()
    return self
  
  def __exit__(self, type, value, traceback):
    self.dt = time.time() - self.start
    self.t += self.dt

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
