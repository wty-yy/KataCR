# -*- coding: utf-8 -*-
'''
@File    : datapath_manager.py
@Time    : 2023/11/09 10:41:40
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
This script defines the `PathManger` class for the CR dataset,
this class is used to recursively reading the file path in the directory
that matches the regular expression.
'''
from katacr.utils.related_pkgs.utility import *
import katacr.build_dataset.constant as const
import re

class PathManager:
  def __init__(self, path_dataset: Path = const.path_dataset):
    self.path = path_dataset
  
  def sample(
      self, subset: str = None,
      part: int | str = None,
      video_name: str = None,
      name: str = None,
      regex: str = ""
    ) -> List[Path]:
    """
    Sample a path from `path_dataset/subset/part/video_name/name`,
    if any key word is `None`, then it will return all the sub files.
    The file names will be filtered by regular expression `regex`.
    """
    path = self.path
    if subset is not None: path = path.joinpath(subset)
    if part is not None: path = path.joinpath("part"+str(part))
    if video_name is not None: path = path.joinpath(video_name)
    if name is not None: path = path.joinpath(name)
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
