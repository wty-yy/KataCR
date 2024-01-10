from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.detection.parser import YOLOv5Args, get_args_and_writer
from katacr.build_dataset.constant import MAX_NUM_BBOXES
from torch.utils.data import Dataset, DataLoader
from katacr.build_dataset.generator import Generator
import cv2
import numpy as np
from PIL import Image
import warnings
import random
from katacr.detection.cfg import train_datasize

from katacr.utils.detection.data import (
  transform_hsv, transform_pad, show_box
)

class YOLODataset(Dataset):
  def __init__(self, image_shape: int, subset: str, path_dataset: Path):
    self.img_shape = image_shape
    self.subset = subset
    self.path_dataset = path_dataset
    self.augment = False if subset == 'val' else True
    self.max_num_box = MAX_NUM_BBOXES
    if subset == 'val':
      path_annotation = self.path_dataset.joinpath(f"annotation.txt")
      paths = np.genfromtxt(str(path_annotation), dtype=np.str_)
      self.paths_img, self.paths_box = paths[:, 0], paths[:, 1]
      self.datasize = len(self.paths_img)
    else:
      self.generator = Generator()
      self.datasize = train_datasize
  
  def __len__(self):
    return self.datasize
  
  @staticmethod
  def _check_bbox_need_placeholder(bboxes):
    if len(bboxes) == 0:
      bboxes = np.array([[0,0,1,1,-1]], dtype=np.float32)  # placeholder
    return bboxes
  
  def load_file(self, idx):
    if self.subset == 'val':
      path_img, path_box = self.paths_img[idx], self.paths_box[idx]
      img = np.array(Image.open(str(self.path_dataset.joinpath(path_img))).convert('RGB')).astype('uint8')
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        box = np.loadtxt(self.path_dataset.joinpath(path_box))
      if len(box):
        box = np.roll(box.reshape(-1, 12), -1, axis=1)  # (x,y,w,h,*states,cls)
      else:
        box = box.reshape(0, 12)
    else:
      self.generator.reset()
      self.generator.add_tower()
      self.generator.add_unit(80)
      img, box = self.generator.build()

    h0, w0 = img.shape[:2]
    if box[:, :4].max() <= 1:  # ratio to pixel
      box[:, [0,2]] *= w0
      box[:, [1,3]] *= h0
    r = min(self.img_shape[0] / h0, self.img_shape[1] / w0)
    if r != 1:  # resize the max aspect to image_size
      interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA  # enlarge or shrink
      img = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
      if len(box):
        box[:, [0,2]] *= img.shape[1] / w0
        box[:, [1,3]] *= img.shape[0] / h0
    return img, box

  def __getitem__(self, idx):
    img, box = self.load_file(idx)

    img, (dh, dw) = transform_pad(img, self.img_shape)
    box[:, 0] += dw
    box[:, 1] += dh
    if self.augment:
      pass
      # img = transform_hsv(img)
      # if random.random() < 0.5:  # Flip left-right
      #   img = np.fliplr(img)
      #   if len(box):
      #     box[:, 0] = img.shape[1] - box[:, 0]
    pbox = np.zeros((self.max_num_box, 12))  # faster than np.pad
    if len(box):
      pbox[:len(box)] = box
    return img.copy(), pbox.copy(), len(box)

class DatasetBuilder:
  args: YOLOv5Args

  def __init__(self, args: YOLOv5Args):
    self.args = args
  
  def get_dataset(self, subset: str = 'val'):
    dataset = YOLODataset(
      image_shape=self.args.image_shape, subset=subset,
      path_dataset=self.args.path_dataset
    )
    ds = DataLoader(
      dataset, batch_size=self.args.batch_size,
      shuffle=subset == 'train',
      num_workers=self.args.num_data_workers,
      drop_last=True,
    )
    return ds

if __name__ == '__main__':
  args = get_args_and_writer(no_writer=True)
  ds_builder = DatasetBuilder(args)
  # args.batch_size = 1
  # ds = ds_builder.get_dataset(subset='train')
  ds = ds_builder.get_dataset(subset='val')
  print("Dataset size:", len(ds))
  iterator = iter(ds)
  # image, bboxes, num_bboxes = next(iterator)
  # image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  # print(image.shape, bboxes.shape, num_bboxes.shape)
  # for image, bboxes, num_bboxes in tqdm(ds):
  #   image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  #   # print(type(image))
  for i in range(2):
    image, bboxes, num_bboxes = next(iterator)
    image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
    print(image.shape, bboxes.shape, num_bboxes.shape)
    show_box(image[0], bboxes[0][np.arange(num_bboxes[0])])
  