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
