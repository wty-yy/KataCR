from ultralytics.engine.results import Results, Boxes, ops, torch, Annotator, deepcopy, LetterBox, colors, Path, LOGGER, save_one_box, np

class CRBoxes(Boxes):
  def __init__(self, boxes, orig_shape) -> None:
    if boxes.ndim == 1:
      boxes = boxes[None, :]
    n = boxes.shape[-1]
    assert n in (7, 8), f"expected 7 or 8 values but got {n}"  # xyxy, track_id, conf, cls, bel TODO
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    self.data = boxes
    self.orig_shape = orig_shape
    self.is_track = n == 8  # TODO
    self.orig_shape = orig_shape

  @property
  def id(self):
    """Return the track IDs of the boxes (if available)."""
    return self.data[:, -4] if self.is_track else None  # TODO

  @property
  def cls(self):
    return self.data[:, -2:]  # TODO

  @property
  def conf(self):
    return self.data[:, -3]  # TODO

class CRResults(Results):
  def __init__(self, orig_img, path, names, boxes=None, logits_boxes=None, masks=None, probs=None, keypoints=None, obb=None) -> None:
    self.orig_img = orig_img
    self.orig_shape = orig_img.shape[:2]
    self.orig_boxes = boxes
    self.logits_boxes = logits_boxes
    self.boxes = CRBoxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
    self.masks = None
    self.probs = None
    self.keypoints = None
    self.obb = None
    self.speed = {"preprocess": None, "inference": None, "postprocess": None}  # milliseconds per image
    self.names = names
    self.path = path
    self.save_dir = None
    self._keys = "boxes", "masks", "probs", "keypoints", "obb"

  def update(self, boxes=None, masks=None, probs=None, obb=None):
    """Update the boxes, masks, and probs attributes of the Results object."""
    if boxes is not None:
      self.boxes = CRBoxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
  
  def plot(
    self,
    conf=True,
    line_width=None,
    font_size=None,
    font="Arial.ttf",
    pil=False,
    img=None,
    im_gpu=None,
    kpt_radius=5,
    kpt_line=True,
    labels=True,
    boxes=True,
    masks=True,
    probs=True,
    show=False,
    save=False,
    filename=None,
  ):
    """
    Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

    Args:
      conf (bool): Whether to plot the detection confidence score.
      line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
      font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
      font (str): The font to use for the text.
      pil (bool): Whether to return the image as a PIL Image.
      img (numpy.ndarray): Plot to another image. if not, plot to original image.
      im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
      kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
      kpt_line (bool): Whether to draw lines connecting keypoints.
      labels (bool): Whether to plot the label of bounding boxes.
      boxes (bool): Whether to plot the bounding boxes.
      masks (bool): Whether to plot the masks.
      probs (bool): Whether to plot classification probability
      show (bool): Whether to display the annotated image directly.
      save (bool): Whether to save the annotated image to `filename`.
      filename (str): Filename to save image to if save is True.

    Returns:
      (numpy.ndarray): A numpy array of the annotated image.

    Example:
      ```python
      from PIL import Image
      from ultralytics import YOLO

      model = YOLO('yolov8n.pt')
      results = model('bus.jpg')  # results list
      for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image
      ```
    """
    if img is None and isinstance(self.orig_img, torch.Tensor):
      img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    annotator = Annotator(
      deepcopy(self.orig_img if img is None else img),
      line_width,
      font_size,
      font,
      pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
      example=names,
    )

    # Plot Segment results
    if pred_masks and show_masks:
      if im_gpu is None:
        img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
        im_gpu = (
          torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
          .permute(2, 0, 1)
          .flip(0)
          .contiguous()
          / 255
        )
      idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
      annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
      for d in reversed(pred_boxes):
        # TODO
        c, bel, conf, id = int(d.cls[0, 0]), int(d.cls[0, 1]), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
        name = ("" if id is None else f"id:{id} ") + names[c] + str(bel)  # TODO: Add bel after name
        label = (f"{name} {conf:.2f}" if conf else name) if labels else None
        box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
        annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

    # Plot Classify results
    if pred_probs is not None and show_probs:
      text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
      x = round(self.orig_shape[0] * 0.03)
      annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

    # Plot Pose results
    if self.keypoints is not None:
      for k in reversed(self.keypoints.data):
        annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

    # Show results
    if show:
      annotator.show(self.path)

    # Save results
    if save:
      annotator.save(filename)

    return annotator.result()
  
  def get_data(self):
    if not isinstance(self.boxes.data, np.ndarray):
      if self.boxes.data.device != 'cpu':
        # xyxy, (track_id), conf, cls, bel
        self.boxes.data = self.boxes.data.cpu().numpy()
      else:
        self.boxes.data = self.boxes.data.numpy()
    return self.boxes.data
  
  def get_rgb(self):
    return self.orig_img[...,::-1]  # uint8, RGB

  def show_box(self, draw_center_point=False, verbose=False, use_overlay=True, show_conf=False, save_path=None, fontsize=12, show_track=True):
    from katacr.utils.detection import plot_box_PIL, build_label2colors
    from katacr.constants.label_list import idx2unit
    from katacr.constants.state_list import idx2state
    from PIL import Image
    img = self.get_rgb()
    # box = self.boxes.data.cpu().numpy()  # xyxy, (track_id), conf, cls, bel
    box = self.get_data()
    img = img.copy()
    if isinstance(img, np.ndarray):
      if img.max() <= 1.0: img *= 255
      img = Image.fromarray(img.astype('uint8'))
    if use_overlay:
      overlay = Image.new('RGBA', img.size, (0,0,0,0))  # build a RGBA overlay
    if len(box):
      label2color = build_label2colors(box[:,-2])
    for b in box:
      bel = int(b[-1])
      cls = int(b[-2])
      conf = float(b[-3])
      text = idx2unit[cls]
      text += idx2state[int(bel)]
      if show_track and box.shape[1] == 8:
        text += ' ' + str(int(b[-4]))
      if show_conf:
        text += ' ' + f'{conf:.2f}'
      plot_box = lambda x: plot_box_PIL(
          x, b[:4],
          text=text,
          box_color=label2color[cls],
          format='voc', draw_center_point=draw_center_point,
          fontsize=fontsize
        )
      if use_overlay: overlay = plot_box(overlay)
      else: img = plot_box(img)
    if use_overlay:
      img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    if verbose:
      img.show()
    if save_path is not None:
      img.save(str(save_path))
    return np.array(img)[...,::-1]  # BGR

  def verbose(self):
    """Return log string for each task."""
    log_string = ""
    probs = self.probs
    boxes = self.boxes
    if len(self) == 0:
      return log_string if probs is not None else f"{log_string}(no detections), "
    if probs is not None:
      log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
    if boxes:
      for c in boxes.cls[:, 0].unique():  # TODO
        n = (boxes.cls[:, 0] == c).sum()  # detections per class
        log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
    return log_string

  def save_txt(self, txt_file, save_conf=False):
    """
    Save predictions into txt file.

    Args:
      txt_file (str): txt file path.
      save_conf (bool): save confidence score or not.
    """
    is_obb = self.obb is not None
    boxes = self.obb if is_obb else self.boxes
    masks = self.masks
    probs = self.probs
    kpts = self.keypoints
    texts = []
    if probs is not None:
      # Classify
      [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
    elif boxes:
      # Detect/segment/pose
      for j, d in enumerate(boxes):
        # TODO: add bel
        c, bel, conf, id = int(d.cls[0, 0]), int(d.cls[0, 1]), float(d.conf), None if d.id is None else int(d.id.item())
        line = (c, bel, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
        if masks:
          seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
          line = (c, *seg)
        if kpts is not None:
          kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
          line += (*kpt.reshape(-1).tolist(),)
        line += (conf,) * save_conf + (() if id is None else (id,))
        texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
      Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
      with open(txt_file, "a") as f:
        f.writelines(text + "\n" for text in texts)

  def save_crop(self, save_dir, file_name=Path("im.jpg")):
    """
    Save cropped predictions to `save_dir/cls/file_name.jpg`.

    Args:
      save_dir (str | pathlib.Path): Save path.
      file_name (str | pathlib.Path): File name.
    """
    if self.probs is not None:
      LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
      return
    if self.obb is not None:
      LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
      return
    for d in self.boxes:
      save_one_box(
        d.xyxy,
        self.orig_img.copy(),
        # TODO: use 0 column
        file=Path(save_dir) / self.names[int(d.cls[0, 0])] / f"{Path(file_name)}.jpg",
        BGR=True,
      )

  def tojson(self, normalize=False):
    """Convert the object to JSON format."""
    if self.probs is not None:
      LOGGER.warning("Warning: Classify task do not support `tojson` yet.")
      return

    import json

    # Create list of detection dictionaries
    results = []
    data = self.boxes.data.cpu().tolist()
    h, w = self.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
      box = {"x1": row[0] / w, "y1": row[1] / h, "x2": row[2] / w, "y2": row[3] / h}
      # TODO (xywh, conf, cls, bel)
      conf = row[-3]
      class_id = int(row[-2])
      bel = int(row[-1])
      name = self.names[class_id]
      result = {"name": name, "class": class_id, "confidence": conf, "box": box, "belong": bel}  # TODO: add bel
      if self.boxes.is_track:
        result["track_id"] = int(row[-4])  # track ID TODO
      if self.masks:
        x, y = self.masks.xy[i][:, 0], self.masks.xy[i][:, 1]  # numpy array
        result["segments"] = {"x": (x / w).tolist(), "y": (y / h).tolist()}
      if self.keypoints is not None:
        x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
        result["keypoints"] = {"x": (x / w).tolist(), "y": (y / h).tolist(), "visible": visible.tolist()}
      results.append(result)

    # Convert detections to JSON
    return json.dumps(results, indent=2)
  