# -*- coding: utf-8 -*-
'''
@File    : datapath_manager.py
@Time    : 2023/11/09 10:41:40
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : 
This script defines the `PathManger` class for the CR dataset,
this class is used to recursively reading the file path in the directory
that matches the regular expression.
'''
from katacr.utils.related_pkgs.utility import *
import katacr.build_dataset.constant as const
import re
import warnings

class PathManager:
  def __init__(self, path_dataset: Path = const.path_dataset):
    self.path = path_dataset
  
  def search(
      self, subset: str = None,
      part: int | str = None,
      video_name: str = None,
      name: str = None,
      regex: str = "",
      drop_regex: str | None = None,
    ) -> List[Path]:
    """
    Sample a path from `path_dataset/subset/part/video_name/name`,
    if any key word is `None`, then it will return all the sub files.
    The file names will be filtered by regular expression `regex` and 
    drop the file names with `drop_regex`.
    """
    path = self.path
    if subset is not None: path /= subset
    if part is not None:
      if isinstance(part, int):
        part = "part"+str(part)
      path /= part
    if video_name is not None: path /= video_name
    if name is not None: path /= str(name)
    if not path.exists():
      # warnings.warn(f"\n### Warning: The sample path `{path}` don't exist, skip it. ###")
      return []
    matcher = re.compile(regex)
    drop_matcher = re.compile(drop_regex) if drop_regex is not None else None
    paths = []
    def dfs(path: Path):
      if path.is_file():
        if drop_matcher is not None and drop_matcher.search(path.name):
          return
        if matcher.search(path.name):
          paths.append(path)
        return
      subpaths = sorted(list(path.iterdir()))
      for path in subpaths: dfs(path)
    dfs(path)
    return paths
