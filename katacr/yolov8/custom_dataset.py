from ultralytics.data import YOLODataset
from ultralytics.data.dataset import load_dataset_cache_file, DATASET_CACHE_VERSION, get_hash, TQDM, LOGGER, LOCAL_RANK, HELP_URL, ThreadPool, NUM_THREADS, repeat, save_dataset_cache_file, Path, torch, cv2, Instances
from katacr.build_dataset.generator import Generator
from katacr.yolov8.cfg import unit_nums, map_update_mode, intersect_ratio_thre, train_datasize, img_size, noise_unit_ratio
from katacr.constants.label_list import idx2unit

class CRDataset(YOLODataset):

  def __init__(self, *args, data=None, task="detect", **kwargs):
    self.img_path = None
    if kwargs['img_path'] is not None:  # val
      kwargs.pop('seed')
      super().__init__(*args, data=data, task=task, **kwargs)
    else:  # train, generation dataset
      self.unit_nums = unit_nums
      self.name_inv = {n: i for i, n in data['names'].items()}
      self.generator = Generator(
        seed=kwargs['seed'],
        intersect_ratio_thre=intersect_ratio_thre,
        map_update={'mode': map_update_mode, 'size': 5},
        avail_names=list(data['names'].values()),
        noise_unit_ratio=noise_unit_ratio)
      self.data = data
      self.augment, self.rect, self.imgsz = kwargs['augment'], kwargs['rect'], kwargs['imgsz']
      self.use_segments = self.use_keypoints = self.use_obb = False
      self.transforms = self.build_transforms(hyp=kwargs['hyp'])
  
  def __len__(self):
    if self.img_path is not None:
      return len(self.labels)
    else:
      return train_datasize
  
  # def __getitem__(self, index):
  #   if self.img_path:
  #     return super().__getitem__(index)
  #   else:
  #     self.generator.reset()
  #     self.generator.add_tower()
  #     self.generator.add_unit(self.unit_nums)
  #     img, box, _ = self.generator.build(box_format='cxcywh', img_size=img_size)
  #     img = np.ascontiguousarray(img.transpose(2, 0, 1))
  #     bboxes = box[:, :4]
  #     cls = np.array([self.name_inv[idx2unit[i]] for i in box[:, 5]], np.int32)  # Convert global idx to local idx
  #     cls = np.stack([cls, box[:, 4]], 1)  # cls, bel
  #     labels = {
  #       'img': torch.from_numpy(img),
  #       'cls': torch.from_numpy(cls),
  #       'bboxes': torch.from_numpy(bboxes),
  #       'batch_idx': torch.zeros(cls.shape[0]),
  #       'im_file': None,
  #     }
  #     return labels
  
  def get_image_and_label(self, index):
    if self.img_path is not None:
      return super().get_image_and_label(index)
    self.generator.reset()
    self.generator.add_tower()
    self.generator.add_unit(self.unit_nums)
    img, box, _ = self.generator.build(box_format='cxcywh', img_size=img_size)
    bboxes = box[:, :4]
    cls = np.array([self.name_inv[idx2unit[i]] for i in box[:, 5]])  # Convert global idx to local idx
    cls = np.stack([cls, box[:, 4]], 1).astype(np.float32)  # cls, bel
    # labels = {
    #   'img': torch.from_numpy(img),
    #   'cls': torch.from_numpy(cls),
    #   'bboxes': torch.from_numpy(bboxes),
    #   'batch_idx': torch.zeros(cls.shape[0]),
    #   'im_file': None,
    # }
    label = {
      'im_file': None,
      'ratio_pad': (1.0, 1.0),
      'rect_shape': np.array(img_size[::-1], np.float32),
      'ori_shape': img.shape[:2],
      'resized_shape': img.shape[:2],
      'cls': cls,
      # 'bboxes': bboxes,
      # 'segments': [],
      # 'keypoints': None,
      # 'normalized': True,
      'bbox_format': 'xywh',
      'img': img[...,::-1],
      'instances': Instances(bboxes, np.zeros((0, 1000, 2), np.float32), None, 'xywh', True),
    }
    return label
  
  # def update_labels_info(self, label):
  #   print(label)
  #   print(list(label.keys()))
  #   exit()
  #   return super().update_labels_info(label)
    
  def get_labels(self):
    """Returns dictionary of labels for YOLO training."""
    def img2label_paths(img_paths):
      return [x.rsplit('.', 1)[0] + '.txt' for x in img_paths]  # TODO: Add label path with image
    self.label_files = img2label_paths(self.im_files)
    prefix_path = "/images/part2/"
    n = self.label_files[0].find(prefix_path) + len(prefix_path)
    cache_path = Path(self.label_files[0][:n] + "yolo_label.cache")
    print(cache_path)
    try:
      cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
      assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
      assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
    except (FileNotFoundError, AssertionError, AttributeError):
    # except Exception as e:
      cache, exists = self.cache_labels(cache_path), False  # run cache ops

    # Display cache
    nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
    if exists and LOCAL_RANK in (-1, 0):
      d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
      TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
      if cache["msgs"]:
        LOGGER.info("\n".join(cache["msgs"]))  # display warnings

    # Read cache
    [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
    labels = cache["labels"]
    if not labels:
      LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
    self.im_files = [lb["im_file"] for lb in labels]  # update im_files

    # Check if the dataset is all boxes or all segments
    lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
    len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
    if len_segments and len_boxes != len_segments:
      LOGGER.warning(
        f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
        f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
        "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
      )
      for lb in labels:
        lb["segments"] = []
    if len_cls == 0:
      LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
    return labels

  def cache_labels(self, path=Path("./labels.cache")):
    """
    Cache dataset labels, check images and read shapes.

    Args:
      path (Path): Path where to save the cache file. Default is Path('./labels.cache').

    Returns:
      (dict): labels.
    """
    x = {"labels": []}
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
    total = len(self.im_files)
    nkpt, ndim = self.data.get("kpt_shape", (0, 0))
    if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
      raise ValueError(
        "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
        "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
      )
    with ThreadPool(NUM_THREADS) as pool:
      results = pool.imap(
        func=verify_image_label,
        iterable=zip(
          self.im_files,
          self.label_files,
          repeat(self.prefix),
          repeat(self.use_keypoints),
          repeat(self.data["names"]),  # TODO: give names
          repeat(nkpt),
          repeat(ndim),
        ),
      )
      pbar = TQDM(results, desc=desc, total=total)
      for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
        nm += nm_f
        nf += nf_f
        ne += ne_f
        nc += nc_f
        if im_file:
          x["labels"].append(
            dict(
              im_file=im_file,
              shape=shape,
              cls=lb[:, 0:2],  # n, 2  TODO: 1->2
              bboxes=lb[:, 2:],  # n, 4
              segments=segments,
              keypoints=keypoint,
              normalized=True,
              bbox_format="xywh",
            )
          )
        if msg:
          msgs.append(msg)
        pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
      pbar.close()

    if msgs:
      LOGGER.info("\n".join(msgs))
    if nf == 0:
      LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
    x["hash"] = get_hash(self.label_files + self.im_files)
    x["results"] = nf, nm, ne, nc, len(self.im_files)
    x["msgs"] = msgs  # warnings
    x["version"] = None  # add cache version
    # save_dataset_cache_file(self.prefix, path, x)  # TODO: Not save label cache, it will make multi-detector wrong label.
    return x

from ultralytics.data.utils import Image, np, exif_size, IMG_FORMATS, ImageOps, segments2boxes, os
def verify_image_label(args):
  """Verify one image-label pair."""
  im_file, lb_file, prefix, keypoint, names, nkpt, ndim = args
  names_inv = {n: i for i, n in names.items()}
  # Number (missing, found, empty, corrupt), message, segments, keypoints
  nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
  # Verify images
  im = Image.open(im_file)
  im.verify()  # PIL verify
  shape = exif_size(im)  # image size
  shape = (shape[1], shape[0])  # hw
  assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
  assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
  if im.format.lower() in ("jpg", "jpeg"):
    with open(im_file, "rb") as f:
      f.seek(-2, 2)
      if f.read() != b"\xff\xd9":  # corrupt JPEG
        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

  # Verify labels
  if os.path.isfile(lb_file):
    nf = 1  # label found
    with open(lb_file) as f:
      lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
      tmp = []
      for x in lb:  # just detect box whose label in detection range
        name = idx2unit[int(x[0])]
        if name in names_inv:
          x[0] = str(names_inv[name])
          tmp.append(x)
      lb = tmp
      classes = np.array([(x[0], x[5]) for x in lb], dtype=np.float32)  # TODO: Add belong class target
      # segments = [np.array(x[1:5], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, bel, xy1...)
      xywh = [np.array(x[1:5], dtype=np.float32) for x in lb]  # TODO: Load xywh directly
      # lb = np.concatenate((classes.reshape(-1, 2), segments2boxes(segments)), 1)  # (cls, bel, xywh)
      lb = np.concatenate((classes, xywh), 1)  # (cls, bel, xywh)
      lb = np.array(lb, dtype=np.float32)
    nl = len(lb)
    if nl:
      assert lb.shape[1] == 6, f"labels require 6 columns, {lb.shape[1]} columns detected"  # TODO: 6 columns
      points = lb[:, 2:]
      assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
      assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

      # All labels
      max_cls = lb[:, 0].max()  # max label count
      assert max_cls <= len(names), (
        f"Label class {int(max_cls)} exceeds dataset class count {len(name)}. "
        f"Possible class labels are 0-{len(names) - 1}"
      )
      _, i = np.unique(lb, axis=0, return_index=True)
      if len(i) < nl:  # duplicate row check
        lb = lb[i]  # remove duplicates
        if segments:
          segments = [segments[x] for x in i]
        msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
    else:
      ne = 1  # label empty
      lb = np.zeros((0, (6 + nkpt * ndim) if keypoint else 5), dtype=np.float32)  # TODO: 6 columns
  else:
    nm = 1  # label missing
    lb = np.zeros((0, (6 + nkpt * ndim) if keypoints else 5), dtype=np.float32)  # TODO: 6 columns
  lb = lb[:, :6]  # TODO: 6 columns
  return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
  # except Exception as e:
  #   nc = 1
  #   msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
  #   return [None, None, None, None, None, nm, nf, ne, nc, msg]
