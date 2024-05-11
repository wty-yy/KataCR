import numpy as np
from katacr.build_dataset.constant import path_dataset
from katacr.policy.offline.dataset import DatasetBuilder
from katacr.utils.detection import build_label2colors
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from katacr.policy.replay_data.data_display import GridDrawer, build_label2colors
from katacr.policy.perceptron.utils import pil_draw_text
from katacr.policy.offline.dataset import BAR_SIZE, N_BAR_SIZE
from katacr.classification.predict import CardClassifier

card_cls = CardClassifier()
idx2cls = card_cls.idx2card

path_dataset = path_dataset / "replay_data/golem_ai/WTY_20240419_golem_ai_episodes_1.npy.xz"
ds_builder = DatasetBuilder(path_dataset, 5)
# ds_builder.debug()
use_card_idx = True
ds = ds_builder.get_dataset(1, 1, use_card_idx=use_card_idx, random_interval=2)
cv2.namedWindow("Arena", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
for s, a, rtg, timestep, y in tqdm(ds):
  for x in [s, a, y]:
    for k, v in x.items():
      x[k] = v.numpy()
  rtg = rtg.numpy(); timestep = timestep.numpy()
  # continue
  # print(s['arena'].shape, s['arena_mask'].shape, s['cards'].shape, s['elixir'].shape)
  # print(a['select'].shape, a['pos'].shape)
  # print(s['arena'].dtype, s['arena_mask'].dtype, s['cards'].dtype, s['elixir'].dtype, a['select'].dtype, a['pos'].dtype)
  print(idx2cls)
  print(s['cards'][0])
  print(a['select'][0])
  print(y['select'][0])
  for i in range(ds_builder.n_step):
    select = a['select'][0,i]
    cards = s['cards'][0,i]
    if (use_card_idx and select != 0) or (not use_card_idx and idx2cls[str(select)] != 'empty'):
      if use_card_idx:
        print(f"Action select={select}, card_name={idx2cls[str(cards[select])]} at frame={i}")
      else:
        print(f"Action select={select}, card_name={idx2cls[str(select)]} at frame={i}")
  for i in range(ds_builder.n_step):
    # img = s['arena'][0,i,...,0]
    arena = s['arena'][0,i]  # (32, 18, [cls, bel, bar1, bar2])
    mask = s['arena_mask'][0,i,...,None]
    select = y['select'][0,i]
    cards = s['cards'][0,i]
    delay = y['delay'][0,i]
    if use_card_idx:
      print(f"Target Action select={select}, card_name={idx2cls[str(cards[select])]}, delay={delay}")
    else:
      print(f"Target Action select={select}, card_name={idx2cls[str(select)]}, delay={delay}")
    # print("RTG:", rtg[0,i])
    # print(mask.shape, img.shape)
    label2color = build_label2colors(arena[...,0].reshape(-1))
    label2color[-1] = (255,255,255)
    drawer = GridDrawer()
    subwindows = []
    for r in range(arena.shape[0]):
      for c in range(arena.shape[1]):
        if mask[r,c]:
          cls = arena[r,c,0]
          bel = arena[r,c,1]
          drawer.paint((c,r), label2color[cls], bel)
          bar1 = arena[r,c,-2*N_BAR_SIZE:-N_BAR_SIZE].reshape(BAR_SIZE[::-1]).astype(np.uint8)
          bar2 = arena[r,c,-N_BAR_SIZE:].reshape(BAR_SIZE[::-1]).astype(np.uint8)
          name = f'bar1 ({c},{r})'
          if bar1.sum() != 0:
            cv2.imshow(name, bar1)
            subwindows.append(name)
          if bar2.sum() != 0:
            name.replace('1', '2')
            cv2.imshow(name, bar2)
            subwindows.append(name)
    # select = a['select'][0,i]
    delay = y['delay'][0,i]
    # if select != 0:
    if delay == 0:
      # y, x = a['pos'][0,i]
      yx = y['pos'][0,i]
      drawer.paint(yx[::-1], (255,236,158), select, rect=False, circle=True, text_color=(0,0,0))
    # img = np.stack(np.vectorize(lambda x: label2color[x])(img), -1).astype(np.uint8)
    # print(mask.shape, img.shape)
    # img = mask * img
    # Image.fromarray(img).show()
    rimg = pil_draw_text(drawer.image, (0,0), f"Frame={i}")
    rimg = np.array(rimg)
    cv2.imshow("Arena", rimg[...,::-1])
    cv2.waitKey(0)
    for w in subwindows:
      cv2.destroyWindow(w)
  # break
