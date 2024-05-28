import scipy.spatial
import bisect, torch, lzma, cv2, random
from io import BytesIO
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from katacr.classification.train import EMPTY_CARD_INDEX
from katacr.utils import colorstr

BAR_SIZE = (24, 8)
BAR_RGB = False
N_BAR_SIZE = np.prod(BAR_SIZE) * (3 if BAR_RGB else 1)

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
      for n in range(len(data['action'])-1, -1, -1):
        if data['action'][n]['card_id'] != 0:  # clip episode terminal to last action frame
          break
      n += 1  # idx -> episode size
      if n < self.n_step:
        print(colorstr('red', 'bold', 'Warning:'), f"Replay buffer {p} length={n} < n_step={self.n_step}, skip it")
        continue
      replay['obs'] += data['state'][:n]
      replay['action'] += data['action'][:n]
      replay['reward'].append(data['reward'][:n])
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
    state = data['obs'] = replay['obs']
    action = data['action'] = replay['action']
    taction = data['target_action'] = [dict() for _ in range(len(action))]
    data['done_idx'] = np.where(replay['terminal'])[0]
    ### Build return-to-go ###
    st, n = -1, len(replay['obs'])
    rtg = data['rtg'] = np.zeros(n, np.float32)
    data['timestep'] = np.array([s['time'] for s in data['obs']], np.int32)
    end_idx = data['end_idx'] = []
    sample_weights = []
    for i in data['done_idx']:
      last_action = None
      for j in range(i, st, -1):
        rtg[j] = replay['reward'][j] + (0 if j == i else rtg[j+1])
        if action[j]['card_id'] != 0:
          last_action = action[j].copy()
          last_action.update({'frame': j, 'card_name_idx': state[j]['cards'][last_action['card_id']]})
        taction[j].update(last_action)
        taction[j]['delay'] = last_action['frame'] - j
        taction[j].pop('frame')
      for j in range(st+1, i+1):
        if j - st >= self.n_step:
        # if i - j + 1 >= self.n_step:
          end_idx.append(j)
          sample_weights.append(action[j]['card_id'] != 0)
      st = i
    data['done_idx'] = np.concatenate([[-1], data['done_idx']])
    self.sample_weights = np.array(sample_weights, np.float32)
    # print("action num:", self.sample_weights.sum())
    # action_ratio = max(self.sample_weights.sum() / len(end_idx), 0.1)  # up to 10%
    action_ratio = self.sample_weights.sum() / len(end_idx)
    self.sample_weights = self.sample_weights * 1 / action_ratio + (1 - self.sample_weights) * 1 / (1 - action_ratio)
    for i in np.where(sample_weights)[0]:
      for j in range(i, min(i+self.n_step, len(self.sample_weights))):
        alpha = 1 / (j - i + 1)
        self.sample_weights[j] = max(self.sample_weights[j], alpha * 1 / action_ratio)
        if replay['terminal'][end_idx[j]]:
          break
    self.action_delays = np.array([a['delay'] for a in taction], np.int32)
    data['end_idx'] = np.array(data['end_idx'], np.int32)
    data['info'] = f"Max rtg: {max(data['rtg']):.2f}, Mean rtg: {np.mean(data['rtg']):.2f}, \
Max timestep: {max(data['timestep'])}, Obs len: {len(data['obs'])}, Datasize: {len(data['end_idx'])}, \
Use action ratio: {action_ratio*100:.2f}%, sample rate: {1/action_ratio:.2f}:{1/(1-action_ratio):.2f}, \
Max action delay: {self.action_delays.max()}, Mean action delay: {self.action_delays.mean():.2f}, \
Action number: {(self.action_delays==0).sum()}"
    # print("Max delay idx:", np.argmax(self.action_delays), "Action frame:", np.where(self.action_delays==0), "Action number:", (self.action_delays==0).sum())
    # print(data['done_idx'])
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
    plt.figure()
    plt.plot(self.action_delays)
    plt.title("Action delay")
    # plt.axis([75, 100, 0, 33])
    plt.show()
  
  def get_dataset(
      self, batch_size: int, num_workers: int = 4, lr_flip: bool = True,
      card_shuffle: bool = True, random_interval: int = 1,
      max_delay=20, use_card_idx=False
    ):
    sample_weights = np.concatenate([self.sample_weights] * (int(lr_flip) + 1))  # origin and flip left and right
    return DataLoader(
      StateActionRewardDataset(
        self.data, n_step=self.n_step, lr_flip=lr_flip,
        card_shuffle=card_shuffle, random_interval=random_interval,
        delay_clip=max_delay, use_card_idx=use_card_idx),
      batch_size=batch_size,
      # shuffle=True,  # use sampler
      persistent_workers=True,
      num_workers=num_workers,
      drop_last=True,
      sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
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

def build_feature(
    state, action, target_action=None,
    lr_flip: bool = False, shuffle: bool = False, shuffle_idx=None,
    train=False, delay_clip=None, use_card_idx=True, empty_card_idx=EMPTY_CARD_INDEX):
  """
  Args:
    state (Dict, from `perceptron.state_builder.get_state()`):
      key=['unit_infos', 'cards', 'elixir']
    action (Dict, from `perceptron.action_builder.get_action()`):
      key=['card_id', 'xy']
    target_action (Dict, nearest action frame, train==True):
      key=['card_id', 'xy', 'delay']
    lr_flip (bool): If taggled, flip arena left and right.
    shuffle (bool): If taggled, the cards will be shuffle by shuffle_idx.
    shuffle_idx (List): Specify shuffle index, otherwise random get one.
    train (bool): If taggled, the target action feature `y` will be returned.
    delay_clip (int): If train, delay of target action will be clip by delay_clip.
    use_card_idx (bool): If taggled, action['select'] will be card_idx, otherwise it is card_name_idx
  Returns:
    s (Dict):
      'arena': Unit features in arena, shape=(32, 18, 386)
      'arena_mask': Mask of arena unit, if there is a unit, it will be taggled, shape=(32, 18)
      'cards': Current cards indexs, shape=(5,)
      'elixir': Current elixir number, shape=()
    a (Dict):
      'select': Index of selecting card, shape=(), value=(0,1,2,3,4)
      'pos': Position (yx) of placing the card, shape=(2,), value in (32x18) and (-1, 0) (padding)
    y (Dict): (If train==True)
      'select': Index of selecting card (future action), shape=(), value=(0,1,2,3)
      'pos': Position (yx) of placing the card (future action), shape=(2,), value in (32x18)
      'delay': Delay time for the nearest future action
  NOTE: Empty index embedding:
    unit_infos['cls'] = None => -1
    s['elixir'] = None => -1
    a['pos'] = None => (0, -1)
  """
  s, a, y = dict(), dict(), dict()
  ### Build State ###
  # cls, bel, bar1, bar2, cards, elixir
  arena = np.zeros((32, 18, 1+1+N_BAR_SIZE*2), np.int32)
  arena_mask = np.zeros((32, 18), np.bool_)
  pos_finder = PositionFinder()
  def cvt_bar(bar):
    if bar is None:
      return np.zeros(N_BAR_SIZE, np.uint8)
    if BAR_RGB:
      ret = cv2.resize(bar, BAR_SIZE).reshape(-1)
    else:
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
  if not use_card_idx:
    if a['select']:
      a['select'] = s['cards'][a['select']]
    else:
      a['select'] = empty_card_idx
  if not train:
    return s, a
  if train and delay_clip is None:
    return s, a, a
  ### Build Target Action ###
  y['select'] = np.array(target_action['card_id'], np.int32)
  if shuffle:
    y['select'] = np.array(np.argwhere(idx == y['select'])[0,0], np.int32) - 1
    assert y['select'] != -1
  if not use_card_idx:
    y['select'] = target_action['card_name_idx']
  xy = np.array(target_action['xy'], np.int32)
  if lr_flip:
    xy[0] = 18 - xy[0] - 1
  y['pos'] = np.array(xy[::-1], np.int32)
  y['delay'] = np.clip(target_action['delay'], 0, delay_clip)
  return s, a, y

class StateActionRewardDataset(Dataset):
  def __init__(
      self, data: dict, n_step: int, lr_flip=True, card_shuffle=True,
      random_interval=1, delay_clip=20, use_card_idx=False):
    self.data, self.n_step = data, n_step
    self.lr_flip = lr_flip
    self.shuffle = card_shuffle
    self.random_interval = random_interval
    self.delay_clip = delay_clip
    self.use_card_idx = use_card_idx
  
  def __len__(self):
    # return np.sum(self.data['timestep']!=0) - self.n_step + 1
    return len(self.data['end_idx']) * (2 if self.lr_flip else 1)
  
  def __getitem__(self, idx):
    datasize = len(self.data['end_idx'])
    lr_flip = idx >= datasize; idx %= datasize
    # print(f"{idx=}, {lr_flip=}")  # DEBUG: flip left and right
    # lr_flip = True
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
    y = {
      'delay': np.empty(L, np.int32),
      'select': np.empty(L, np.int32),
      'pos': np.empty((L, 2), np.int32),
    }
    rtg = np.empty(L, np.float32)
    timestep = np.empty(L, np.int32)
    shuffle_idx = get_shuffle_idx() if self.shuffle else None
    now = done_idx
    pre_done = self.data['done_idx'][bisect.bisect_left(self.data['done_idx'], done_idx)-1]
    idxs = []
    for i in range(L-1, -1, -1):
      ns, na, ny = build_feature(
        data['obs'][now], data['action'][now], data['target_action'][now], lr_flip=lr_flip, shuffle=self.shuffle,
        shuffle_idx=shuffle_idx, train=True, delay_clip=self.delay_clip,
        use_card_idx=self.use_card_idx)
      for x, nx in zip([s, a, y], [ns, na, ny]):
        for k in nx.keys():
          x[k][i] = nx[k]
      rtg[i] = data['rtg'][now]
      timestep[i] = data['timestep'][now]
      # print(idx-pre_done, idx)
      if data['action'][now-1]['card_id']:  # don't skip action frame
        interval = 1
      else:
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
    return s, a, rtg, timestep, y

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
  import os
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # allocate GPU memory as needed
  from katacr.build_dataset.constant import path_dataset
  # path_dataset = path_dataset / "replay_data/golem_ai"
  path_dataset = path_dataset / "replay_data/golem_ai/WTY_20240419_golem_ai_episodes_1.npy.xz"
  # path_dataset = "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai"
  # path_dataset = "/home/yy/Coding/GitHub/KataCR/logs/offline/test_replay_data"
  # path_dataset = "/home/yy/Coding/datasets/Clash-Royale-Dataset/replay_data/golem_ai"
  # path_dataset = "/home/yy/Coding/datasets/Clash-Royale-Dataset/replay_data/golem_ai/WTY_20240419_golem_ai_episodes_1.npy.xz"
  # ds_builder = DatasetBuilder(path_dataset, 30)
  ds_builder = DatasetBuilder(path_dataset, 50)
  # debug_save_features("/home/yy/Coding/GitHub/KataCR/logs/intercation/video1_dataset_50")
  # exit()
  print("n_cards:", ds_builder.n_cards)
  # ds_builder.debug()
  from katacr.utils.detection import build_label2colors
  from PIL import Image
  ds = ds_builder.get_dataset(32, 8)
  for s, a, rtg, timestep, y in tqdm(ds):
    for x in [s, a, y]:
      for k, v in x.items():
        x[k] = v.numpy()
    rtg = rtg.numpy(); timestep = timestep.numpy()
    continue
    print(s['arena'].shape, s['arena_mask'].shape, s['cards'].shape, s['elixir'].shape)
    print(a['select'].shape, a['pos'].shape)
    print(y['select'].shape, y['pos'].shape, y['delay'].shape)
    print(s['arena'].dtype, s['arena_mask'].dtype, s['cards'].dtype, s['elixir'].dtype, a['select'].dtype, a['pos'].dtype, y['delay'].dtype)
    break
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
