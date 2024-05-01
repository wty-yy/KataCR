import scipy.spatial
import bisect, torch, lzma, cv2
from io import BytesIO
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from katacr.utils import colorstr

BAR_SIZE = (24, 8)
N_BAR_SIZE = np.prod(BAR_SIZE)

class DatasetBuilder:
  def __init__(self, path_dataset: str, n_step: int, seed=42):
    torch.manual_seed(seed)
    self.path_dataset = path_dataset
    self.n_step = n_step
    self.n_cards = 0
    self._preload()
  
  def _load_replay(self):
    ### Concatenate all replay buffers ###
    replay = {'obs': [], 'action': [], 'reward': [], 'terminal': []}
    print("Loading replay buffers...", end='')
    path_files = list(Path(self.path_dataset).rglob('*.xz'))
    for p in tqdm(path_files, ncols=80):
      bytes = lzma.open(str(p)).read()
      data = np.load(BytesIO(bytes), allow_pickle=True).item()
      n = len(data['state'])
      if n < self.n_step:
        print(colorstr('red', 'bold', 'Warning:'), f"Replay buffer {p} length={n} < n_step={self.n_step}, skip it")
        continue
      replay['obs'] += data['state']
      replay['action'] += data['action']
      replay['reward'].append(data['reward'])
      t = np.zeros(n, np.bool_)
      t[-1] = True
      replay['terminal'].append(t)
      for s in data['state']:
        self.n_cards = max(self.n_cards, max(s['cards']) + 1)
    replay['reward'] = np.concatenate(replay['reward'])
    replay['terminal'] = np.concatenate(replay['terminal'])
    return replay
  
  def _preload(self):
    data = self.data = {}
    replay = self.replay = self._load_replay()
    # print(len(replay['obs']), len(replay['action']), len(replay['reward']), len(replay['terminal']))
    data['obs'], data['action'] = replay['obs'], replay['action']
    data['done_idx'] = np.where(replay['terminal'])[0]
    ### Build return-to-go ###
    st, n = -1, len(replay['obs'])
    rtg = data['rtg'] = np.zeros(n, np.float32)
    data['timestep'] = np.array([s['time'] for s in data['obs']], np.int32)
    start_idx = data['start_idx'] = []
    for i in data['done_idx']:
      for j in range(i, st, -1):
        rtg[j] = replay['reward'][j] + (0 if j == i else rtg[j+1])
        if i - j + 1 >= self.n_step:
          start_idx.append(j)
      st = i
    data['start_idx'] = np.array(data['start_idx'], np.int32)
    data['info'] = f"Max rtg: {max(data['rtg']):.2f}, Mean rtg: {np.mean(data['rtg']):.2f}, \
Max timestep: {max(data['timestep'])}, Obs len: {len(data['obs'])}, Datasize: {len(data['start_idx'])}"
    print(colorstr("INFO"), "Dataset:", data['info'])
  
  def debug(self):
    import matplotlib.pyplot as plt
    plt.hist(self.data['rtg'])
    plt.title(f"Distribution of Return-To-Go")
    plt.show()
  
  def get_dataset(self, batch_size: int, num_workers: int = 4):
    return DataLoader(
      StateActionRewardDataset(self.data, n_step=self.n_step),
      batch_size=batch_size,
      shuffle=True,
      persistent_workers=True,
      num_workers=num_workers,
      drop_last=True,
    )

class PositionFinder:
  def __init__(self, r=32, c=18):
    self.used = np.zeros((r, c), np.bool_)
    self.center = np.swapaxes(np.array(np.meshgrid(np.arange(r), np.arange(c))), 0, -1) + 0.5  # (r, c, 2)
  
  def find_near_pos(self, xy):
    yx = np.array(xy)[::-1]
    y, x = yx.astype(np.int32)
    if self.used[y, x]:
      avail_center = self.center[~self.used]
      map_index = np.argwhere(~self.used)
      dis = scipy.spatial.distance.cdist(yx.reshape(1, 2), avail_center)
      y, x = map_index[np.argmin(dis)]
    self.used[y, x] = True
    return np.array((x, y), np.int32)

def build_feature(state, action, lr_flip: bool=False):
  """
  Args:
    state (Dict, from `perceptron.state_builder.get_state()`):
      key=['unit_infos', 'cards', 'elixir']
    action (Dict, from `perceptron.action_builder.get_action()`):
      key=['card_id', 'xy']
    lr_flip (bool): If taggled, flip arena left and right.
  Returns:
    s (Dict):
      'arena': Unit features in arena, shape=(32, 18, 386)
      'arena_mask': Mask of arena unit, if there is a unit, it will be taggled, shape=(32, 18)
      'cards': Current cards indexs, shape=(5,)
      'elixir': Current elixir number, shape=()
    a (Dict):
      'select': Index of selecting card, shape=()
      'pos': Position (yx) of placing the card, shape=(2,)
  NOTE: Empty index embedding:
    unit_infos['cls'] = None => -1
    s['elixir'] = None => -1
    a['pos'] = None => (0, -1)
  """
  s, a = dict(), dict()
  ### Build State ###
  # cls, bel, bar1, bar2, cards, elixir
  arena = np.zeros((32, 18, 1+1+N_BAR_SIZE*2), np.int32)
  arena_mask = np.zeros((32, 18), np.bool_)
  pos_finder = PositionFinder()
  def cvt_bar(bar):
    if bar is None:
      return np.zeros(N_BAR_SIZE, np.uint8)
    ret = cv2.cvtColor(cv2.resize(bar, BAR_SIZE), cv2.COLOR_RGB2GRAY).reshape(-1)
    return ret
  for info in state['unit_infos']:
    xy = pos_finder.find_near_pos(info['xy'])
    if lr_flip:
      xy[0] = 18 - xy[0] - 1
    pos = arena[xy[1],xy[0]]
    pos[0] = info['cls'] if info['cls'] is not None else -1
    pos[1] = -1 if info['bel'] == 0 else 1
    pos[-2*N_BAR_SIZE:-N_BAR_SIZE] = cvt_bar(info['bar1'])
    pos[-N_BAR_SIZE:] = cvt_bar(info['bar2'])
    arena_mask[xy[1],xy[0]] = True
  s['arena'] = arena; s['arena_mask'] = arena_mask
  s['cards'] = np.array(state['cards'], np.int32)
  elixir = state['elixir'] if state['elixir'] is not None else -1
  s['elixir'] = np.array(elixir, np.int32)
  ### Build Action ###
  a['select'] = np.array(action['card_id'], np.int32)
  xy = action['xy'] if action['xy'] is not None else (-1, 0)
  a['pos'] = np.array(xy[::-1], np.int32)
  return s, a

class StateActionRewardDataset(Dataset):
  def __init__(self, data: dict, n_step: int, lr_flip=True):
    self.data, self.n_step = data, n_step
    self.lr_flip = lr_flip
  
  def __len__(self):
    # return np.sum(self.data['timestep']!=0) - self.n_step + 1
    return len(self.data['start_idx']) * (2 if self.lr_flip else 1)
  
  def __getitem__(self, idx):
    datasize = len(self.data['start_idx'])
    lr_flip = idx >= datasize; idx %= datasize
    data, n_step = self.data, self.n_step
    # done_idx = idx + n_step - 1
    # bisect_left(a, x): if x in a, return left x index, else return index with elem bigger than x
    # done_idx = min(data['done_idx'][bisect.bisect_left(data['done_idx'], done_idx)], done_idx)
    # idx = done_idx - n_step + 1
    idx = self.data['start_idx'][idx]; done_idx = idx + n_step - 1
    L = self.n_step
    s = {
      'arena': np.empty((L, 32, 18, 1+1+2*N_BAR_SIZE), np.int32),
      'arena_mask': np.empty((L, 32, 18), np.bool_),
      'cards': np.empty((L, 5), np.int32),
      'elixir': np.empty(L, np.int32),
    }
    a = {
      'select': np.empty(L, np.int32),
      'pos': np.empty((L, 2), np.int32),
    }
    for i in range(idx, done_idx+1):
      ns, na = build_feature(data['obs'][i], data['action'][i], lr_flip=lr_flip)
      for x, nx in zip([s, a], [ns, na]):
        for k in x.keys():
          x[k][i-idx] = nx[k]
    rtg = data['rtg'][idx:done_idx+1].astype(np.float32)
    timestep = data['timestep'][idx:done_idx+1].astype(np.int32)
    return s, a, rtg, timestep

if __name__ == '__main__':
  path_dataset = "/home/yy/Coding/datasets/Clash-Royale-Dataset/replay_data"
  ds_builder = DatasetBuilder(path_dataset, 30)
  print("n_cards:", ds_builder.n_cards)
  # ds_builder.debug()
  from katacr.utils.detection import build_label2colors
  from PIL import Image
  ds = ds_builder.get_dataset(32)
  for s, a, rtg, timestep in tqdm(ds):
    for x in [s, a]:
      for k, v in x.items():
        x[k] = v.numpy()
    rtg = rtg.numpy(); timestep = timestep.numpy()
    continue
    print(s['arena'].shape, s['arena_mask'].shape, s['cards'].shape, s['elixir'].shape)
    print(a['select'].shape, a['pos'].shape)
    print(s['arena'].dtype, s['arena_mask'].dtype, s['cards'].dtype, s['elixir'].dtype, a['select'].dtype, a['pos'].dtype)
    img = s['arena'][0,0,...,0]
    label2color = build_label2colors(img.reshape(-1))
    img = np.vectorize(lambda x: label2color[x])(img)
    print(len(img))
    img = np.array(img, np.uint8).transpose([1,2,0])
    print(img.shape)
    Image.fromarray(img).show()
    # cv2.imshow("test", s[...,0])
    # print(rtg.shape)
    # print(timestep.shape)
    break
