# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/11/20 15:57:48
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
Load CR Dataset.
'''
from katacr.utils.related_pkgs.utility import *
from katacr.yolo.parser import YOLOArgs, get_args_and_writer
from torch.utils.data import Dataset, DataLoader
from katacr.build_dataset.constant import MAX_NUM_BBOXES
from katacr.build_dataset.utils.datapath_manager import PathManager
import albumentations as A
import cv2
import numpy as np
from PIL import Image
import warnings

class YOLODataset(Dataset):
  args: YOLOArgs
  shuffle: bool
  no_resize: bool
  path_manager: PathManager
  path_bboxes: List[Path]
  max_num_bboxes: int
  place_holder: np.ndarray

  def __init__(
      self, args: YOLOArgs, shuffle: bool, transform: Callable
    ):
    self.args, self.shuffle, self.transform = (
      args, shuffle, transform
    )
    self.place_holder = np.array([[0,0,1,1,*([-1]*8)]], dtype=np.float32)

    self.max_num_bboxes = MAX_NUM_BBOXES
    self.path_manager = PathManager(self.args.path_dataset)
    self.path_bboxes = self.path_manager.sample(subset='images', part=2, regex=r'^\d+.txt') * self.args.repeat
  
  def __len__(self):
    return len(self.path_bboxes)
  
  def _check_bbox_need_placeholder(self, bboxes):
    if len(bboxes) == 0:
      bboxes = self.place_holder
    return bboxes
  
  def _load_data(self, index):
    path_bboxes = self.path_bboxes[index]
    path_image = path_bboxes.parent.joinpath(path_bboxes.name[:-3]+'jpg')
    image = np.array(Image.open(path_image).convert("RGB"))
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      bboxes = np.loadtxt(path_bboxes)
    if len(bboxes):
      # bboxes parameters: (x, y, w, h, state*7, class_id) Shape: (N,12)
      bboxes = np.roll(bboxes.reshape(-1, 12), -1, axis=1)
    bboxes = self._check_bbox_need_placeholder(bboxes)
    bboxes = bboxes[(bboxes[:,2]>0)&(bboxes[:,3]>0)]  # avoid 0 wide and height
    return image, bboxes
  
  def __getitem__(self, index):
    image, bboxes = self._load_data(index)
    if self.transform:
      try:
        transformed = self.transform(image=image, bboxes=bboxes)
      except Exception as e:
        print("Error Id:", self.path_bboxes[index])
        raise e
      image, bboxes = transformed['image'], np.array(transformed['bboxes'])
    # Maybe remove all the bboxes after transform
    bboxes = self._check_bbox_need_placeholder(bboxes)
    # Convert output to yolo format with pixel size!
    bboxes[:,0] *= image.shape[1]
    bboxes[:,1] *= image.shape[0]
    bboxes[:,2] *= image.shape[1]
    bboxes[:,3] *= image.shape[0]
    num_bboxes = np.sum(bboxes[:,4] != -1)
    bboxes = np.concatenate([
      bboxes,
      np.repeat(
        self.place_holder,
        repeats=self.max_num_bboxes-bboxes.shape[0], axis=0
    )], axis=0)
    return image, bboxes, num_bboxes

class DatasetBuilder:
  args: YOLOArgs

  def __init__(self, args: YOLOArgs):
    self.args = args
  
  def get_transform(self, subset: str):
    train_transform = A.Compose(
      [
        A.Resize(height=self.args.image_shape[0], width=self.args.image_shape[1]),
        A.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.5, hue=0.015, p=0.4),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
      ],
      bbox_params=A.BboxParams(format='yolo')
    )
    val_transform = A.Compose(
      [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
      ],
      bbox_params=A.BboxParams(format='yolo')
    )
    return train_transform if subset=='train' else val_transform
  
  def get_dataset(self, subset='train', shuffle: bool = True):
    dataset = YOLODataset(
      self.args, shuffle, self.get_transform(subset)
    )
    ds = DataLoader(
      dataset, batch_size=self.args.batch_size,
      shuffle=True,
      num_workers=self.args.num_data_workers,
      drop_last=True,
    )
    return ds

def show_bbox(image, bboxes, draw_center_point=False):
  """
  Show the image with bboxes use PIL.

  Args:
    image: Shape=(H,W,3) or (H,W)
    bboxes: Shape=(N,12), last dim means: (x,y,w,h,states*7,label)
    draw_center_point: Whether to draw the center point of all the bboxes
  """
  from katacr.utils.detection import plot_box_PIL, build_label2color
  from katacr.constants.label_list import idx2unit
  from katacr.constants.state_list import idx2state
  if type(image) != Image.Image:
    image = Image.fromarray((image*255).astype('uint8'))
  if len(bboxes):
    label2color = build_label2color(range(200))  # same color
    # label2color = build_label2color(bboxes[:,11])
  for bbox in bboxes:
    unitid = int(bbox[11])
    text = idx2unit[unitid] + idx2state[int(bbox[4])]
    for i in range(5, 11):
      if bbox[i] != 0:
        text += ' ' + idx2state[int((i-4)*10 + bbox[i])]
    image = plot_box_PIL(image, bbox[:4], text=text, box_color=label2color[unitid], format='yolo', draw_center_point=draw_center_point)
    # print(label, label2name[label], label2color[label])
  image.show()

if __name__ == '__main__':
  args = get_args_and_writer(no_writer=True)
  ds_builder = DatasetBuilder(args)
  args.batch_size = 3
  ds = ds_builder.get_dataset()
  print("Dataset size:", len(ds))
  iterator = iter(ds)
  # image, bboxes, num_bboxes = next(iterator)
  # image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  # print(image.shape, bboxes.shape, num_bboxes.shape)
  for image, bboxes, num_bboxes in tqdm(ds):
    image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  # for i in range(3):
  #   image, bboxes, num_bboxes = next(iterator)
  #   image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  #   # print(image.shape, bboxes.shape, num_bboxes.shape)
  #   show_bbox(image[0], bboxes[0][np.arange(num_bboxes[0])])
  