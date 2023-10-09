from PIL import Image
import numpy as np
from pathlib import Path

bbox_params = {
    'part1': {
        'time': (0.835, 0.074, 0.165, 0.025),
        'hp0':  (0.166, 0.180, 0.090, 0.020),
        'hp1':  (0.755, 0.183, 0.090, 0.020),
        'hp2':  (0.515, 0.073, 0.090, 0.020),
        'hp3':  (0.162, 0.617, 0.090, 0.020),
        'hp4':  (0.756, 0.617, 0.090, 0.020),
        'hp5':  (0.511, 0.753, 0.090, 0.020),
    },
    'part2': (0.021, 0.073, 0.960, 0.700),
    'part3': (0.000, 0.821, 1.000, 0.179)
}

def extract_bbox(image, x, y, w, h):
    """
    - `(x, y)`: The left top proportion point of the whole image.
    - `(w, h)`: The width and height of the proportion the whole image.
    """
    shape = image.shape
    if len(shape) == 2:
        image = image[...,None]
    x, y = int(shape[1] * x), int(shape[0] * y)
    w, h = int(shape[1] * w), int(shape[0] * h)
    image = image[y:y+h, x:x+w, :]
    if len(shape) == 2: image = image[..., 0]
    return image

def to_gray(image):
    return np.array(Image.fromarray(image).convert('L'))

def process_part1(image):
    images = {}
    for key, value in bbox_params['part1'].items():
        images[key] = extract_bbox(image, *value)
    return images

def process_part2(image):
    image = extract_bbox(image, *bbox_params['part2'])
    return image

def process_part3(image):
    image = extract_bbox(image, *bbox_params['part3'])
    return image

if __name__ == '__main__':
    path_logs = Path("../logs")

    # image = Image.open(str(path_logs.joinpath("start_frame.jpg")))
    # image = Image.open(str(path_logs.joinpath("show_king_tower_hp.jpg")))
    # image = Image.open(str(path_logs.joinpath("start_setting_behind_king_tower.jpg")))
    image = Image.open(str(path_logs.joinpath("end_setting_behind_king_tower.jpg")))
    image = np.array(image)

    part1 = process_part1(image)
    for key, value in part1.items():
        Image.fromarray(value).save(f"part1_{key}.jpg")
    part2 = process_part2(image)
    Image.fromarray(part2).save("part2.jpg")
    part3 = process_part3(image)
    Image.fromarray(part3).save("part3.jpg")

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(5,20))
    # plt.imshow(image, cmap='gray')
    # plt.show()
