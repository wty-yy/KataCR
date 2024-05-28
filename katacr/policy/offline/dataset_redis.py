"""
This is a test, had been deprecated.
"""
import scipy.spatial
import bisect, torch, lzma, cv2, random
from io import BytesIO
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from katacr.utils import colorstr
import redis, pickle

BAR_SIZE = (24, 8)
N_BAR_SIZE = np.prod(BAR_SIZE)

dbs = {
  'obs': 0,
  'action': 1,
  'rtg': 2,
  'timestep': 3,
  'end_idx': 4,
}

class RedisManager:
  def __init__(self, name):
    self.r = redis.Redis(host='localhost', port=6379, db=dbs[name])
    self.idx = 0
  
  def save(self, data):
    self.r.set(str(self.idx), pickle.dumps(data))
    self.idx += 1
  
  def get(self, idx):
    return pickle.loads(self.r.get(str(idx)))
  
  def save_list(self, l):
    for e in l:
      self.save(e)

class DatasetBuilder:
  def __init__(self, path_dataset: str, n_step: int, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    self.path_dataset = path_dataset
    self.n_step = n_step
    self.n_cards = 0
    self._preload()
  
  def _load_replay(self):
    ### Concatenate all replay buffers ###
    replay = {'obs': [], 'action': [], 'reward': [], 'terminal': []}
    print("Loading replay buffers...", end='')
    if Path(self.path_dataset).is_dir():
      path_files = list(Path(self.path_dataset).rglob('*.xz'))
    else:
      path_files = [self.path_dataset]
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
    data = self.data = {name: RedisManager(name) for name in ['obs', 'action', 'rtg', 'timestep']}
    replay = self.replay = self._load_replay()
    # print(len(replay['obs']), len(replay['action']), len(replay['reward']), len(replay['terminal']))
    data['obs'].save_list(replay['obs'])
    data['action'].save_list(replay['action'])
    timestep = [s['time'] for s in replay['obs']]
    data['timestep'].save_list(timestep)
    data['done_idx'] = np.where(replay['terminal'])[0]
    ### Build return-to-go ###
    st, n = -1, len(replay['obs'])
    rtg = np.zeros(n, np.float32)
    end_idx = data['end_idx'] = []
    sample_weights = []
    for i in data['done_idx']:
      for j in range(i, st, -1):
        rtg[j] = replay['reward'][j] + (0 if j == i else rtg[j+1])
      for j in range(st+1, i+1):
        if j - st >= self.n_step:
        # if i - j + 1 >= self.n_step:
          end_idx.append(j)
          sample_weights.append(replay['action'][j]['card_id'] != 0)
      st = i
    data['rtg'].save_list(rtg)
    data['done_idx'] = np.concatenate([[-1], data['done_idx']])
    self.sample_weights = np.array(sample_weights, np.float32)
    # action_ratio = max(self.sample_weights.sum() / len(end_idx), 0.1)  # up to 10%
    action_ratio = self.sample_weights.sum() / len(end_idx)
    self.sample_weights = self.sample_weights * 1 / action_ratio + (1 - self.sample_weights) * 1 / (1 - action_ratio)
    for i in np.where(sample_weights)[0]:
      # print(i)
      for j in range(i, min(i+30, len(self.sample_weights))):
        alpha = 1 / (j - i + 1)
        self.sample_weights[j] = max(self.sample_weights[j], alpha * 1 / action_ratio)
        # print(j, alpha, alpha * 1 / action_ratio)
        if replay['terminal'][j]: break
    data['end_idx'] = np.array(data['end_idx'], np.int32)
    data['info'] = f"Max rtg: {max(rtg):.2f}, Mean rtg: {np.mean(rtg):.2f}, \
Max timestep: {max(timestep)}, Data len: {len(rtg)}, Datasize: {len(data['end_idx'])}, \
Use action ratio: {action_ratio*100:.2f}%, sample rate: {1/action_ratio:.2f}:{1/(1-action_ratio):.2f}"
    print(colorstr("INFO"), "Dataset:", data['info'])
  
  def debug(self):
    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(self.data['rtg'])
    # plt.title("Distribution of Return-To-Go")
    plt.figure()
    plt.plot(self.data['end_idx'], self.sample_weights)
    for i in self.data['end_idx']:
      if self.data['action'][i]['card_id']:
        plt.plot([i, i], [0, 30], 'r--')
    plt.title("Sample ratio")
    # plt.axis([75, 100, 0, 33])
    plt.show()
  
  def get_dataset(
      self, batch_size: int, num_workers: int = 4, lr_flip: bool = True,
      card_shuffle: bool = True, random_interval: int = 2
    ):
    sample_weights = np.concatenate([self.sample_weights] * (int(lr_flip) + 1))  # origin and flip left and right
    return DataLoader(
      StateActionRewardDataset(
        self.data, n_step=self.n_step, lr_flip=lr_flip,
        card_shuffle=card_shuffle, random_interval=random_interval),
      batch_size=batch_size,
      # shuffle=True,  # use sampler
      persistent_workers=True,
      num_workers=num_workers,
      drop_last=True,
      sampler=WeightedRandomSampler(sample_weights, len(sample_weights))
    )

class PositionFinder:
  def __init__(self, r=32, c=18):
    self.used = np.zeros((r, c), np.bool_)
    self.center = np.swapaxes(np.array(np.meshgrid(np.arange(r), np.arange(c))), 0, -1) + 0.5  # (r, c, 2)
  
  def find_near_pos(self, xy):
    yx = np.array(xy)[::-1]
    y, x = yx.astype(np.int32)
    y = np.clip(y, 0, 31)
    x = np.clip(x, 0, 17)
    if self.used[y, x]:
      avail_center = self.center[~self.used]
      map_index = np.argwhere(~self.used)
      dis = scipy.spatial.distance.cdist(yx.reshape(1, 2), avail_center)
      y, x = map_index[np.argmin(dis)]
    self.used[y, x] = True
    return np.array((x, y), np.int32)

def get_shuffle_idx():
  idx = list(range(1, 5))
  random.shuffle(idx)
  idx = np.array([0] + idx, np.int32)
  return idx

def build_feature(state, action, lr_flip: bool = False, shuffle: bool = False, shuffle_idx=None):
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
  if action['xy'] is not None:
    xy = np.array(action['xy'], np.int32)
    if lr_flip:
      xy[0] = 18 - xy[0] - 1
  else:
    xy = (-1, 0)
  a['pos'] = np.array(xy[::-1], np.int32)
  if shuffle:
    if shuffle_idx is not None:
      idx = shuffle_idx
    else:
      idx = get_shuffle_idx()
    # print("before:", s['cards'], a['select'], idx)
    s['cards'] = s['cards'][idx]
    a['select'] = np.array(np.argwhere(idx == a['select'])[0,0], np.int32)
    # print("after: ", s['cards'], a['select'], idx)
  return s, a

class StateActionRewardDataset(Dataset):
  def __init__(self, data: dict, n_step: int, lr_flip=True, card_shuffle=True, random_interval=2):
    self.data, self.n_step = data, n_step
    self.redis = {name: RedisManager(name) for name in ['obs', 'action', 'rtg', 'timestep']}
    self.lr_flip = lr_flip
    self.shuffle = card_shuffle
    self.random_interval = random_interval
  
  def __len__(self):
    # return np.sum(self.data['timestep']!=0) - self.n_step + 1
    return len(self.data['end_idx']) * (2 if self.lr_flip else 1)
  
  def __getitem__(self, idx):
    datasize = len(self.data['end_idx'])
    lr_flip = idx >= datasize; idx %= datasize
    data, L = self.data, self.n_step
    # done_idx = idx + n_step - 1
    # bisect_left(a, x): if x in a, return left x index, else return index with elem bigger than x
    # done_idx = min(data['done_idx'][bisect.bisect_left(data['done_idx'], done_idx)], done_idx)
    # idx = done_idx - n_step + 1
    done_idx = self.data['end_idx'][idx]; idx = done_idx - L + 1
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
    rtg = np.empty(L, np.float32)
    timestep = np.empty(L, np.int32)
    if self.shuffle:
      shuffle_idx = get_shuffle_idx()
    now = done_idx
    pre_done = self.data['done_idx'][bisect.bisect_left(self.data['done_idx'], done_idx)-1]
    idxs = []
    for i in range(L-1, -1, -1):
      ns, na = build_feature(self.redis['obs'].get(now), self.redis['action'].get(now), lr_flip=lr_flip, shuffle=self.shuffle, shuffle_idx=shuffle_idx)
      for x, nx in zip([s, a], [ns, na]):
        for k in x.keys():
          x[k][i] = nx[k]
      rtg[i] = self.redis['rtg'].get(now)
      timestep[i] = self.redis['timestep'].get(now)
      # print(idx-pre_done, idx)
      interval = random.randint(1, min(self.random_interval, idx-pre_done))
      idxs.append(now)
      now -= interval
      idx -= interval - 1
    # for i in range(idx, done_idx+1):
    #   ns, na = build_feature(data['obs'][i], data['action'][i], lr_flip=lr_flip, shuffle=self.shuffle, shuffle_idx=shuffle_idx)
    #   for x, nx in zip([s, a], [ns, na]):
    #     for k in x.keys():
    #       x[k][i-idx] = nx[k]
    # rtg = data['rtg'][idx:done_idx+1].astype(np.float32)
    # timestep = data['timestep'][idx:done_idx+1].astype(np.int32)
    return s, a, rtg, timestep

def debug_save_features(path_save):
  ds = StateActionRewardDataset(ds_builder.data, 30, lr_flip=False)
  s, a, r, t = ds[-1]
  data = {'s': s, 'a': a, 'rtg': r, 'timestep': t}
  # for i in range(len(data)):
  #   if isinstance(data[i], dict):
  #     for k in data[i]:
  #       data[i][k] = data[i][k].numpy()
  #   else:
  #     data[i] = data[i].numpy()
  np.save(path_save, data, allow_pickle=True)

if __name__ == '__main__':
  # path_dataset = "/home/yy/Coding/datasets/Clash-Royale-Dataset/replay_data"
  path_dataset = "/home/yy/Coding/datasets/Clash-Royale-Dataset/replay_data/golem_ai/WTY_20240419_golem_ai_episodes_1.npy.xz"
  ds_builder = DatasetBuilder(path_dataset, 30)
  # ds_builder.debug()
  # debug_save_features("/home/yy/Coding/GitHub/KataCR/logs/intercation/video1_dataset_50")
  # exit()
  print("n_cards:", ds_builder.n_cards)
  # ds_builder.debug()
  from katacr.utils.detection import build_label2colors
  from PIL import Image
  ds = ds_builder.get_dataset(32, 4)
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
