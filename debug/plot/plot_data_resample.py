from katacr.policy.offline.dataset import DatasetBuilder
from katacr.build_dataset.constant import path_dataset
import matplotlib.pyplot as plt
import numpy as np
config = {
    "font.family": ['serif', 'SimSun'], # 衬线字体, 宋体
    "figure.figsize": (16, 4),  # 图像大小
    "font.size": 20, # 字号大小
    "mathtext.fontset": 'cm', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

class PlotDatasetBuilder(DatasetBuilder):
  def debug(self):
    # Sample ratio
    fig, ax = plt.subplots(figsize=(16,4))
    x = self.data['end_idx']
    y = self.sample_weights
    # mask = (614 <= x) & (x <= 710)
    xl, xr = 550, 650
    mask = (xl <= x) & (x <= xr)
    x, y = x[mask], y[mask]
    axis = [x.min()-1, x.max()+1, y.min(), y.max()]
    for i in self.data['end_idx']:  # Draw action
      if self.data['action'][i]['card_id']:
        if i in x:
          l2, = ax.plot([i, i], [0, y.max()], '--', lw=3, color='tab:red', label="动作执行")
        print(i)
    l1, = ax.plot(x, y, lw=3)
    y = self.action_delays[self.n_step-1:][mask]
    y = np.clip(y, 0, 20)
    axis[2] = min(axis[2], y.min()-0.5)
    axis[3] = max(axis[3], y.max())
    l3, = ax.plot(x, y, 'o', lw=3)
    ax.set_xlim(*axis[:2]); ax.set_ylim(*axis[2:])
    xticks = np.arange(axis[0]+1, axis[1]+1, 10)
    xticks = sorted(list(range(xl//10*10, xr//10*10, 10)) + [xl, xr])
    ax.set_xticks(xticks)
    ax.set_xlabel("采样轨迹/帧")
    # ax.set_xlabel("Sampled trajactory / frame")
    # ax.set_ylabel("frame")
    fig.legend([l1,l2,l3], ['重采样频次', '动作执行帧', '动作预测值'], loc='lower center', ncols=3, frameon=False)
    # fig.legend([l1,l2,l3], ['Resampling frequency', 'Action execution frame', 'Action prediction value'], loc='lower center', ncols=3, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)
    plt.savefig("resample_and_delay.pdf")
    plt.show()
  
path_dataset = path_dataset / "replay_data/golem_ai/WTY_20240419_golem_ai_episodes_1.npy.xz"
ds_builder = PlotDatasetBuilder(path_dataset, n_step=50)
ds_builder.debug()