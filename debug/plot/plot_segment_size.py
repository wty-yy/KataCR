import yaml
from pathlib import Path
from katacr.yolov8.cfg import detection_range, max_detect_num
from katacr.build_dataset.constant import path_dataset
from katacr.constants.label_list import idx2unit, unit2idx
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
config = {
    "font.family": ['serif', 'SimSun'], # 宋体,衬线字体
    "figure.figsize": (14, 6),  # 图像大小
    "font.size": 20, # 字号大小
    "mathtext.fontset": 'cm', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

invalid_units = ['selected', 'text', 'mirror', 'tesla-evolution-shock', 'zap-evolution']

path_root = Path(__file__).parents[2]
path_logs = path_root / 'logs'
path_logs.mkdir(exist_ok=True)
path_seg_dataset = path_dataset / "images/segment"

class Ploter:

  def __init__(self):
    self._preload_segment_size()
  
  def _preload_segment_size(self):
    us = {}
    for p in tqdm(list(path_seg_dataset.glob("*"))):
      if p.is_dir():
        if p.name in unit2idx:
          sz = us[p.name] = []
          for img in p.glob("*.png"):
            sz.append(np.prod(Image.open(img).size))
          if len(sz) == 0 or p.name in invalid_units:
            us.pop(p.name)
          else:
            us[p.name] = np.mean(sz)
    self.unit_size = sorted(us.items(), key=lambda x: x[1])
  
  def plot_segment_size(self):
    us = self.unit_size
    xs = np.arange(len(us)) + 1
    ys = np.array([x[1] for x in us], np.float32)
    fig, ax = plt.subplots(figsize=(18,6))
    # plt.plot(xs, ys, 'o-', label='切片面积')
    plt.plot(xs, ys, 'o-', label='Sliced area')
    # plt.xlabel("切片编号（按切片面积从小到大排序）")
    plt.xlabel("Slice number (sorted by slice area)")
    xticks = np.array(sorted(list(range(0, 150, 20)[1:]) + [1, 10, 30, 50, 75, 150]), np.int32)
    plt.xticks(xticks)
    # plt.ylabel("平均切片面积（单位：像素$\\times 10^4$）", fontdict={'fontfamily': 'SimSun'})
    plt.ylabel("Average slice area (unit: pixels $\\times 10^4$)")
    yticks = list(range(0, int(9.6e4), int(2e4))) + [int(9.6e4)]
    plt.yticks(yticks, [0,2,4,6,8,9.6])
    axis = [xs.min()-0.5, xs.max()+0.5, 0, ys.max()+1300]

    # Model
    double_xs = list(range(0, len(xs), len(xs) // 2))[1:]
    # plt.plot(double_xs*2, [axis[-2],axis[-1]], '--', lw=2, label="双模型切片分割线")
    plt.plot(double_xs*2, [axis[-2],axis[-1]], '--', lw=2, label="Dual-model separator")
    tri_xs = list(range(0, len(xs), len(xs) // 3))[1:]
    # plt.plot(tri_xs[0:1]*2+tri_xs[1:2]*2, [axis[-2],axis[-1]+100,axis[-1]+100,axis[-2]], '--', lw=2, label="三模型切片分割线")
    plt.plot(tri_xs[0:1]*2+tri_xs[1:2]*2, [axis[-2],axis[-1]+100,axis[-1]+100,axis[-2]], '--', lw=2, label="Triple-model separator")
    # plt.plot()

    # Highlight
    highlight = list(xticks[:-1])
    highlight.remove(75)
    highlight = np.array(highlight)
    # plt.plot(highlight, ys[highlight-1], 'r*', ms=10, label="切片可视化采样点")
    plt.plot(highlight, ys[highlight-1], 'r*', ms=10, label="Slice sampling points")
    print("Highlight units:", [us[x-1][0] for x in highlight])
    # print(us)

    plt.legend()
    plt.axis(axis)
    plt.tight_layout()
    # plt.savefig("./segment_size.png", dpi=300)
    plt.savefig("./segment_size.svg")
    plt.show()

if __name__ == '__main__':
  ploter = Ploter()
  ploter.plot_segment_size()
