from ultralytics.models.yolo.detect.train import DetectionTrainer, copy, torch_distributed_zero_first, LOGGER, build_dataloader
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data import YOLODataset
from ultralytics.utils import colorstr, RANK
from pathlib import Path
from katacr.yolov8.custom_model import CRDetectionModel
from katacr.yolov8.custom_validator import CRDetectionValidator
from katacr.yolov8.custom_utils import plot_images

class CRTrainer(DetectionTrainer):

  def get_model(self, cfg=None, weights=None, verbose=True):
    """Return a YOLO detection model."""
    model = CRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
    if weights:
      model.load(weights)
    return model
  
  def get_validator(self):
    """Returns a DetectionValidator for YOLO model validation."""
    self.loss_names = "box_loss", "cls_loss", "dfl_loss"
    return CRDetectionValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )

  def build_dataset(self, img_path, mode="train", batch=None):
    return CRDataset(
      img_path=img_path,
      imgsz=self.args.imgsz,
      cache=self.args.cache,
      augment=False,
      prefix=colorstr(f"{mode} 123: "),
      # rect=self.args.rect,
      rect=True,  # TODO: set rect True, since CR Dataset is same size
      batch_size=batch,
      stride=32,
      pad=0.0,
      single_cls=False,
      classes=None,  # only include class
      fraction=1.0,

      data=self.data,
    )
  
  def plot_training_samples(self, batch, ni):
    plot_images(
      images=batch["img"],
      batch_idx=batch["batch_idx"],
      cls=batch["cls"],  # TODO: cls with 2 columns, (B, 2)
      bboxes=batch["bboxes"],
      paths=batch["im_file"],
      fname=self.save_dir / f"train_batch{ni}.jpg",
      on_plot=self.on_plot,
      names=self.data['names'],  # TODO: add names
    )
  
  def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
    """Construct and return dataloader."""
    assert mode in ["train", "val"]
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
      dataset = self.build_dataset(dataset_path, mode, batch_size)
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle:
      # LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
      LOGGER.info("OK ✅ 'rect=True' is compatible with CR DataLoader shuffle, setting shuffle=Talse")
      shuffle = True
    workers = self.args.workers if mode == "train" else self.args.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

from ultralytics.data.dataset import load_dataset_cache_file, DATASET_CACHE_VERSION, get_hash, TQDM, LOGGER, LOCAL_RANK, HELP_URL, ThreadPool, NUM_THREADS, repeat, save_dataset_cache_file
class CRDataset(YOLODataset):
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
          repeat(len(self.data["names"])),
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
    save_dataset_cache_file(self.prefix, path, x)
    return x

from ultralytics.data.utils import Image, np, exif_size, IMG_FORMATS, ImageOps, segments2boxes, os
def verify_image_label(args):
  """Verify one image-label pair."""
  im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
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
      if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
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
      assert max_cls <= num_cls, (
        f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
        f"Possible class labels are 0-{num_cls - 1}"
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

