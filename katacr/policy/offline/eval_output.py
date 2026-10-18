"""Output directory policy shared by offline evaluation entry points."""

import time
from pathlib import Path


def resolve_output_dir(root, vid_path=None, save=True, timestamp=None):
  """Return an evaluation output directory, or ``None`` when saving is off."""
  if not save:
    return None

  stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
  root = Path(root)
  if vid_path is not None:
    return root / "logs" / "offline_eval" / Path(vid_path).stem / stamp
  return root / "logs" / "interaction" / stamp
