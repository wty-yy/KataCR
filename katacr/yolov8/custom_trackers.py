from ultralytics.trackers.track import check_yaml, IterableSimpleNamespace, yaml_load, partial, torch, Path
from ultralytics.trackers.byte_tracker import BYTETracker, STrack, matching, TrackState, np, xywh2ltwh
from ultralytics.trackers.bot_sort import BOTSORT

class CRSTrack(STrack):
  def __init__(self, xywh, score, cls, bel):
    """Initialize new STrack instance."""
    super().__init__(xywh, score, cls)
    self.bel = bel  # TODO

  def re_activate(self, new_track, frame_id, new_id=False):
    """Reactivates a previously lost track with a new detection."""
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_track.tlwh)
    )
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
        self.track_id = self.next_id()
    self.score = new_track.score
    self.cls = new_track.cls
    self.bel = new_track.bel  # TODO
    self.angle = new_track.angle
    self.idx = new_track.idx

  @property
  def result(self):
    """Get current tracking results."""
    coords = self.xyxy if self.angle is None else self.xywha
    return coords.tolist() + [self.track_id, self.score, self.cls, self.bel, self.idx]  # TODO

  def update(self, new_track, frame_id):
    """
    Update the state of a matched track.

    Args:
        new_track (STrack): The new track containing updated information.
        frame_id (int): The ID of the current frame.
    """
    self.frame_id = frame_id
    self.tracklet_len += 1

    new_tlwh = new_track.tlwh
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_tlwh)
    )
    self.state = TrackState.Tracked
    self.is_activated = True

    self.score = new_track.score
    self.cls = new_track.cls
    self.bel = new_track.bel  # TODO
    self.angle = new_track.angle
    self.idx = new_track.idx

class CRBYTETracker(BYTETracker):
  def update(self, results, img=None):
    """Updates object tracker with new detections and returns tracked object bounding boxes."""
    self.frame_id += 1
    activated_stracks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []

    scores = results.conf
    bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
    # Add index
    bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
    cls = results.cls[:,0]
    bel = results.cls[:,1]

    remain_inds = scores > self.args.track_high_thresh
    inds_low = scores > self.args.track_low_thresh
    inds_high = scores < self.args.track_high_thresh

    inds_second = np.logical_and(inds_low, inds_high)
    dets_second = bboxes[inds_second]
    dets = bboxes[remain_inds]
    scores_keep = scores[remain_inds]
    scores_second = scores[inds_second]
    cls_keep = cls[remain_inds]
    cls_second = cls[inds_second]
    bel_keep = bel[remain_inds]
    bel_second = bel[inds_second]

    detections = self.init_track(dets, scores_keep, cls_keep, bel_keep, img)
    # Add newly detected tracklets to tracked_stracks
    unconfirmed = []
    tracked_stracks = []  # type: list[STrack]
    for track in self.tracked_stracks:
      if not track.is_activated:
        unconfirmed.append(track)
      else:
        tracked_stracks.append(track)
    # Step 2: First association, with high score detection boxes
    strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
    # Predict the current location with KF
    self.multi_predict(strack_pool)
    if hasattr(self, "gmc") and img is not None:
      warp = self.gmc.apply(img, dets)
      STrack.multi_gmc(strack_pool, warp)
      STrack.multi_gmc(unconfirmed, warp)

    dists = self.get_dists(strack_pool, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

    for itracked, idet in matches:
      track = strack_pool[itracked]
      det = detections[idet]
      if track.state == TrackState.Tracked:
        track.update(det, self.frame_id)
        activated_stracks.append(track)
      else:
        track.re_activate(det, self.frame_id, new_id=False)
        refind_stracks.append(track)
    # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
    detections_second = self.init_track(dets_second, scores_second, cls_second, bel_second, img)
    r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
    # TODO
    dists = matching.iou_distance(r_tracked_stracks, detections_second)
    matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
    for itracked, idet in matches:
      track = r_tracked_stracks[itracked]
      det = detections_second[idet]
      if track.state == TrackState.Tracked:
        track.update(det, self.frame_id)
        activated_stracks.append(track)
      else:
        track.re_activate(det, self.frame_id, new_id=False)
        refind_stracks.append(track)

    for it in u_track:
      track = r_tracked_stracks[it]
      if track.state != TrackState.Lost:
        track.mark_lost()
        lost_stracks.append(track)
    # Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections = [detections[i] for i in u_detection]
    dists = self.get_dists(unconfirmed, detections)
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    for itracked, idet in matches:
      unconfirmed[itracked].update(detections[idet], self.frame_id)
      activated_stracks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
      track = unconfirmed[it]
      track.mark_removed()
      removed_stracks.append(track)
    # Step 4: Init new stracks
    for inew in u_detection:
      track = detections[inew]
      if track.score < self.args.new_track_thresh:
        continue
      track.activate(self.kalman_filter, self.frame_id)
      activated_stracks.append(track)
    # Step 5: Update state
    for track in self.lost_stracks:
      if self.frame_id - track.end_frame > self.max_time_lost:
        track.mark_removed()
        removed_stracks.append(track)

    self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
    self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
    self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
    self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    self.removed_stracks.extend(removed_stracks)
    if len(self.removed_stracks) > 1000:
      self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum
    
    return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

  def init_track(self, dets, scores, cls, bel, img=None):
    """Initialize object tracking with detections and scores using STrack algorithm."""
    return [CRSTrack(xyxy, s, c, b) for (xyxy, s, c, b) in zip(dets, scores, cls, bel)] if len(dets) else []  # detections

class CRBOTSORT(BOTSORT, CRBYTETracker):
  ...
  
TRACKER_MAP = {"bytetrack": CRBYTETracker, "botsort": CRBOTSORT}

def on_predict_start(predictor: object, persist: bool = False) -> None:
  """
  Initialize trackers for object tracking during prediction.

  Args:
    predictor (object): The predictor object to initialize trackers for.
    persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

  Raises:
    AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
  """
  if hasattr(predictor, "trackers") and persist:
    return

  tracker = check_yaml(predictor.args.tracker)
  cfg = IterableSimpleNamespace(**yaml_load(tracker))

  if cfg.tracker_type not in ["bytetrack", "botsort"]:
    raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

  trackers = []
  for _ in range(predictor.dataset.bs):
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
    trackers.append(tracker)
  predictor.trackers = trackers

def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
  """
  Postprocess detected boxes and update with object tracking.

  Args:
    predictor (object): The predictor object containing the predictions.
    persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
  """
  bs = predictor.dataset.bs
  path, im0s = predictor.batch[:2]

  is_obb = predictor.args.task == "obb"
  for i in range(bs):
    if not persist and predictor.vid_path[i] != str(predictor.save_dir / Path(path[i]).name):  # new video
      predictor.trackers[i].reset()

    det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
    if len(det) == 0:
      continue
    tracks = predictor.trackers[i].update(det, im0s[i])
    if len(tracks) == 0:
      continue
    # idx = tracks[:, -1].astype(int)  # TODO: WHY?
    # predictor.results[i] = predictor.results[i][idx]

    update_args = dict()
    update_args["obb" if is_obb else "boxes"] = torch.as_tensor(tracks[:, :-1])
    predictor.results[i].update(**update_args)

def register_tracker(model: object, persist: bool) -> None:
  """
  Register tracking callbacks to the model for object tracking during prediction.

  Args:
    model (object): The model object to register tracking callbacks for.
    persist (bool): Whether to persist the trackers if they already exist.
  """
  model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
  model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))

def cr_on_predict_start(detector, persist: bool = True) -> None:
  if detector.tracker is not None and persist:
    return

  tracker = check_yaml(detector.tracker_cfg_path)
  cfg = IterableSimpleNamespace(**yaml_load(tracker))

  if cfg.tracker_type not in ["bytetrack", "botsort"]:
    raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

  detector.tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)

def cr_on_predict_postprocess_end(detector, persist: bool = True) -> None:
  """
  Postprocess detected boxes and update with object tracking.

  Args:
      predictor (object): The predictor object containing the predictions.
      persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
  """
  det = detector.result.boxes.cpu().numpy()
  if len(det) == 0:
    return
  tracks = detector.tracker.update(det)
  if len(tracks) == 0:
    return
  # idx = tracks[:, -1].astype(int)  # TODO: WHY need this
  # predictor.results[i] = predictor.results[i][idx]

  update_args = dict(boxes=torch.as_tensor(tracks[:, :-1]))
  detector.result.update(**update_args)