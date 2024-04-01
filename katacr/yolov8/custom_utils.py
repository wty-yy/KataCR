from ultralytics.utils.plotting import threaded, np, torch, math, cv2, Annotator, Path, ops, colors, contextlib
@threaded
def plot_images(
    images,
    batch_idx,
    cls,  # TODO: (B, 2) last dim=(cls, bel)
    bboxes=np.zeros(0, dtype=np.float32),
    confs=None,
    masks=np.zeros(0, dtype=np.uint8),
    kpts=np.zeros((0, 51), dtype=np.float32),
    paths=None,
    fname="images.jpg",
    names=None,
    on_plot=None,
    max_subplots=16,
    save=True,
    conf_thres=0.25,
):
    """Plot image grid with labels."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920 * 4  # max image size TODO: BIGGER * 4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths[i]:  # BUG: paths is [None, None, ...]
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")  # TODO: (idxs, 2)
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., 0::2] *= w  # scale to pixels
                        boxes[..., 1::2] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0::2] += x
                boxes[..., 1::2] += y
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j, 0]  # TODO: (idxs, 2)
                    bel = classes[j, 1]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}{bel}" if labels else f"{c}{bel} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)

from ultralytics.utils.ops import xywh2xyxy, torch, xywh2xyxy, time, nms_rotated, torchvision, LOGGER
def non_max_suppression(
  prediction,
  conf_thres=0.25,
  iou_thres=0.45,
  classes=None,
  agnostic=False,
  multi_label=False,
  labels=(),
  max_det=300,
  nc=0,  # number of classes (optional)
  max_time_img=0.05,
  max_nms=30000,
  max_wh=7680,
  in_place=True,
  rotated=False,
):
  """
  Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

  Args:
    prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
      containing the predicted boxes, classes, and masks. The tensor should be in the format
      output by a model, such as YOLO.
    conf_thres (float): The confidence threshold below which boxes will be filtered out.
      Valid values are between 0.0 and 1.0.
    iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
      Valid values are between 0.0 and 1.0.
    classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
    agnostic (bool): If True, the model is agnostic to the number of classes, and all
      classes will be considered as one.
    multi_label (bool): If True, each box may have multiple labels. (TODO: The multi_label class must be replace by maximum confidence class in NMS)
    labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
      list contains the apriori labels for a given image. The list should be in the format
      output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
    max_det (int): The maximum number of boxes to keep after NMS.
    nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
    max_time_img (float): The maximum time (seconds) for processing one image.
    max_nms (int): The maximum number of boxes into torchvision.ops.nms().
    max_wh (int): The maximum box width and height in pixels.
    in_place (bool): If True, the input prediction tensor will be modified in place.

  Returns:
    (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
      shape (num_boxes, 7 + num_masks) containing the kept boxes, with columns
      (x1, y1, x2, y2, confidence, class, belong, mask1, mask2, ...).
  """

  # Checks
  assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
  assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
  if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
    prediction = prediction[0]  # select only inference output

  bs = prediction.shape[0]  # batch size
  nc = nc or (prediction.shape[1] - 4)  # number of classes
  nm = prediction.shape[1] - nc - 4
  mi = 4 + nc  # mask start index
  xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

  # Settings
  # min_wh = 2  # (pixels) minimum box width and height
  time_limit = 2.0 + max_time_img * bs  # seconds to quit after
  multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

  prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
  if not rotated:
    if in_place:
      prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    else:
      prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

  t = time.time()
  output = [torch.zeros((0, 7 + nm), device=prediction.device)] * bs  # TODO: default last dim=7
  for xi, x in enumerate(prediction):  # image index, image inference
    # Apply constraints
    # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
    x = x[xc[xi]]  # confidence

    # Cat apriori labels if autolabelling
    if labels and len(labels[xi]) and not rotated:
      lb = labels[xi]
      v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
      v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
      v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
      x = torch.cat((x, v), 0)

    # If none remain process next image
    if not x.shape[0]:
      continue

    # Detections matrix nx6 (xyxy, conf, cls, bel)  # TODO
    # box, cls, mask = x.split((4, nc, nm), 1)
    box, cls, bel, mask = x.split((4, nc-1, 1, nm), 1)

    if multi_label:  # BUG: It's useless, since nms will replace lower conf with best conf class
      raise RuntimeError("multi_label is useless")
      i, j = torch.where(cls > conf_thres)
      x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
    else:  # best class only
      # cls: (N, 200), conf: (N, 1), j: (N, 1) in (0, 199)
      # bel: (N, 1)
      conf, j = cls.max(1, keepdim=True)
      bel = (bel > 0.5).float()
      x = torch.cat((box, conf, j.float(), bel), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if classes is not None:
      x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
      continue
    if n > max_nms:  # excess boxes
      x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    scores = x[:, 4]  # scores
    if rotated:
      boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
      i = nms_rotated(boxes, scores, iou_thres)
    else:
      boxes = x[:, :4] + c  # boxes (offset by class)
      i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    i = i[:max_det]  # limit detections

    # # Experimental
    # merge = False  # use merge-NMS
    # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
    #   # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
    #   from .metrics import box_iou
    #   iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
    #   weights = iou * scores[None]  # box weights
    #   x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
    #   redundant = True  # require redundant detections
    #   if redundant:
    #     i = i[iou.sum(1) > 1]  # require redundancy

    output[xi] = x[i]
    if (time.time() - t) > time_limit:
      LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
      break  # time limit exceeded

  return output