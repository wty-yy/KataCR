from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

path_eval_name = [
  'DT_4L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240519_224135',
  'StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step30__0__20240516_125734',
  'StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240513_114808',
  'StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step100__0__20240513_114949',
  'StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step100__0__20240516_124617',
  'StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step30__0__20240512_181548',
  'StARformer_3L_v0.8_golem_ai_step50',
  'StARformer_3L_pred_cls_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240516_125201',
  'StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252'
]
path_root = Path(__file__).parents[2]
path_logs = path_root / 'logs'
path_logs.mkdir(exist_ok=True)
path_eval_dir = path_root / "logs/interaction"
path_eval_dirs = [path_eval_dir / name for name in path_eval_name]
config = {
    "font.family": 'serif, SimSun', # 衬线字体
    "figure.figsize": (14, 6),  # 图像大小
    "font.size": 16, # 字号大小
    "mathtext.fontset": 'cm', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)
translation = {
  # 'survival_time': '对局时长',
  'survival_time': 'Match duration',
  # 'total_reward': '总奖励',
  'total_reward': 'Total reward',
  # 'use_actions': '动作数',
  'use_actions': 'Number of actions',
}

class Ploter:
  def __init__(self, path_eval_dirs=path_eval_dirs):
    self._preload(path_eval_dirs)

  @staticmethod
  def get_name(data):
    name = f"{data['model_name']}_{data['model_length']}L_step{data['model_nstep']}"
    if data['pred_cls']:
      name += '_pred_cls'
    if not data['delay']:
      name += '_no_delay'
    return name
  
  def _preload(self, paths):
    data = self.data = []
    for p_dir in sorted(paths):
      p_dir = Path(p_dir)
      p_name = p_dir.name
      model_name = p_name.split('_')[0]
      model_length = int(p_name[p_name.find('L_')-1])
      model_nstep = int(p_name[p_name.find('_step')+5:].split('_')[0])
      delay = not ('no_delay' in p_name)
      pred_cls = 'pred_cls' in p_name
      d = dict(model_name=model_name, model_length=model_length, model_nstep=model_nstep, pred_cls=pred_cls, delay=delay)
      data.append(d)
      eval = d['eval'] = {'ep': []}
      for key in translation:
        eval[key] = {'mu': [], 'sigma': [], 'min': [], 'max': []}
      eval['winrate'] = []
      for p_ep in sorted(p_dir.glob('*'), key=lambda p: int(p.name.split('_load')[1])):
        if p_ep.is_dir():
          ep = int(p_ep.name.split('_load')[1])
          p_csv = next(p_ep.glob('*.csv'))
          df = pd.read_csv(p_csv)
          if sum(df['total_reward'] > 0):
            success = df.loc[df['total_reward']>0]
            print(f"{self.get_name(d)} epoch {ep} in episode {list(success['episode'])} win!")
          eval['ep'].append(ep)
          x = df[~df['model_name'].isna()][list(translation.keys())]
          winrate = (x['total_reward'] > 0).mean()
          mean, std, mn, mx = x.mean(), x.std(), x.min(), x.max()
          # assert np.isnan(x.iloc[1]), f"{p_csv} don't have avg row!"
          for name in translation:
            eval[name]['mu'].append(mean[name])
            eval[name]['sigma'].append(std[name])
            eval[name]['min'].append(mn[name])
            eval[name]['max'].append(mx[name])
          eval['winrate'].append(winrate)
            
    # print(data)
  
  def plot(self, model_names=None, colors=None, filename="policy_evaluation", ncols=3, subplot_bottom=0.25):
    if model_names is None: model_names = self.model_names
    if colors is None: colors = list(mcolors.TABLEAU_COLORS.values())[:len(model_names)]
    fig, axs = plt.subplots(1,3,figsize=(12,5))
    for ax, key in zip(axs, translation.keys()):
      axis = [np.inf, -np.inf] * 2
      for name, color in zip(model_names, colors):
        for d in self.data:
          if self.get_name(d) != name: continue
          eval = d['eval']
          name = self.get_name(d)
          # label = name.replace('_', ' ').replace('pred cls', '（预测全体类别）').replace('no delay', '（离散预测）')
          label = name.replace('_', ' ').replace('pred cls', '(Predict all classes)').replace('no delay', '(Discrete actions)')
          x = np.array(eval['ep'], np.int32)
          y = np.array(eval[key]['mu'], np.float32)
          mx = np.array(eval[key]['max'], np.float32)
          mn = np.array(eval[key]['min'], np.float32)
          std = np.array(eval[key]['sigma'], np.float32) * 1.0
          flag = x <= 10
          x, y, std, mx, mn = x[flag], y[flag], std[flag], mx[flag], mn[flag]
          for i, a in zip([0,2], [x,y]):
            axis[i] = min(axis[i], a.min())
            axis[i+1] = max(axis[i+1], a.max())
          line, = ax.plot(x, y, label=label, lw=3, color=color)
          ax.fill_between(x, y-std, y+std, color=line.get_color(), alpha=0.2)
          # ax.fill_between(x, mn, mx, color=line.get_color(), alpha=0.2)
        ax.set_xlim(*axis[:2])
        # ax.set_ylim(*axis[2:])
        xticks = list(np.arange(0,axis[1],1)+1)
        ax.set_xticks(xticks)
        # ax.set_xlabel("训练回合")
        ax.set_xlabel("Epoch")
        # ax.set_title(translation[key])
        ax.set_ylabel(translation[key])
    for d in self.data:
      if self.get_name(d) in model_names:
        eval = d['eval']
        r = np.array(eval['total_reward']['mu'], np.float32)
        winrate = np.array(eval['winrate'], np.float32)
        idx = np.argmax(r + winrate * 100)
        def get_best(x):
          x = x.copy()
          for k, v in x.items():
            if isinstance(v, list): x[k] = v[idx]
            else: x[k] = get_best(v)
          return x
        print("Model", self.get_name(d))
        print(get_best(eval))
        print("="*10)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncols=ncols, frameon=False)
    # fig.tight_layout(w_pad=0)
    fig.tight_layout(w_pad=1)
    fig.subplots_adjust(bottom=subplot_bottom)
    plt.savefig(filename+".pdf")
    # plt.savefig(filename+".svg")
    plt.show()
  
  @property
  def model_names(self):
    names = []
    for d in self.data:
      names.append(self.get_name(d))
    return names

if __name__ == '__main__':
  ploter = Ploter()
  print("model_names:", ploter.model_names)
  ploter.plot()
  ploter.plot(
    ['DT_4L_step50', 'StARformer_2L_step50_no_delay', 'StARformer_2L_step30', 'StARformer_3L_step50'],
    colors=['tab:blue', 'tab:green', 'tab:orange', 'tab:red'],
    filename="diff_model",
    ncols=2, subplot_bottom=0.3
  )
  ploter.plot(
    ['StARformer_2L_step30', 'StARformer_2L_step50', 'StARformer_2L_step100'],
    colors=['tab:blue', 'tab:green', 'tab:red'],
    filename="diff_2L_nstep"
  )
  ploter.plot(
    ['StARformer_3L_step30', 'StARformer_3L_step50', 'StARformer_3L_step100'],
    colors=['tab:blue', 'tab:red', 'tab:green'],
    filename="diff_3L_nstep"
  )