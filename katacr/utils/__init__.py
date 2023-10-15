from PIL import Image
import numpy as np

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
