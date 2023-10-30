from katacr.utils.related_pkgs.utility import *
import re

class PathManager:
  
  def __init__(self, path_dataset: Path):
    self.path = path_dataset
  
  def sample(
      self, subset: str = None,
      part: int = None,
      video_name: str = None,
      last_names: Sequence = (),
      regex: str = ""
    ) -> List[Path]:
    """
    Sample a path from `path_dataset/subset/part/video_name/*last_names`,
    if any key word is `None`, then it will return all the sub files.
    The file names will be filtered by regular expression `regex`.
    """
    path = self.path
    if subset is not None: path.joinpath(subset)
    if part is not None: path.joinpath("part"+str(part))
    if video_name is not None: path.joinpath(video_name)
    if len(last_names) != 0:
      for name in last_names: path.joinpath(name)
    if not path.exists(): raise Exception(f"The sample path `{path}` don't exist.")
    matcher = re.compile(regex)
    paths = []
    def dfs(path: Path):
      if path.is_file():
        if matcher.search(path.name):
          paths.append(path)
        return
      subpaths = sorted(list(path.iterdir()))
      for path in subpaths: dfs(path)
    dfs(path)
    return paths
