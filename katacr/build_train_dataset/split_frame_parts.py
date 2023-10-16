from PIL import Image
import numpy as np
from pathlib import Path
import katacr.build_train_dataset.constant as const

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
    for key, value in const.split_bbox_params['part1'].items():
        images[key] = extract_bbox(image, *value)
    return images

def process_part2(image):
    image = extract_bbox(image, *const.split_bbox_params['part2'])
    return image

def process_part3(image):
    image = extract_bbox(image, *const.split_bbox_params['part3'])
    return image

def process_part4(image):
    images = {}
    for key, value in const.split_bbox_params['part4'].items():
        images[key] = extract_bbox(image, *value)
    return images

if __name__ == '__main__':
    path_logs = const.path_logs
    path_extract = path_logs.joinpath("extract_frames")
    # path_frame = path_extract.joinpath("OYASSU_20230201")
    path_frame = path_extract.joinpath("OYASSU_20230211")
    # path_frame = path_extract.joinpath("OYASSU_20210528")

    # image = Image.open(str(path_logs.joinpath("start_frame.jpg")))
    # image = Image.open(str(path_logs.joinpath("show_king_tower_hp.jpg")))
    # image = Image.open(str(path_logs.joinpath("start_setting_behind_king_tower.jpg")))
    image = Image.open(str(path_frame.joinpath("end_episode1.jpg")))
    image = np.array(image)
    print("Image shape:", image.shape)

    path_image_save = path_logs.joinpath("split_image")
    path_image_save.mkdir(exist_ok=True)
    part1 = process_part1(image)
    for key, value in part1.items():
        Image.fromarray(value).save(str(path_image_save.joinpath(f"part1_{key}.jpg")))
    part2 = process_part2(image)
    Image.fromarray(part2).save(str(path_image_save.joinpath("part2.jpg")))
    part3 = process_part3(image)
    Image.fromarray(part3).save(str(path_image_save.joinpath("part3.jpg")))
    part4 = process_part4(image)
    for key, value in part4.items():
        Image.fromarray(value).save(str(path_image_save.joinpath(f"part4_{key}.jpg")))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,20))
    # plt.imshow(image)
    plt.imshow(part4['mid'])
    plt.show()
