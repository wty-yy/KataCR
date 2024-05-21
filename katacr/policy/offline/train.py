import os
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # allocate GPU memory as needed
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.16'  # allocate GPU memory as needed
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.32'  # allocate GPU memory as needed
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.24'  # allocate GPU memory as needed
# 14.345Gb <-> (bs32,step30) (batch size, n_step)
# 28.785Gb <-> (bs16,step100), (bs32,step50) (batch size, n_step)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))
from katacr.policy.offline.parse_and_logs import parse_args_and_writer, logs
from katacr.policy.offline.dataset import DatasetBuilder
from katacr.utils.ckpt_manager import CheckpointManager
from tqdm import tqdm
import numpy as np
from katacr.constants.label_list import unit_list

def train():
  ### Parse augment and TF Writer ###
  args, writer = parse_args_and_writer()
  ### Dataset ###
  ds_builder = DatasetBuilder(args.replay_dataset, args.n_step)
  if 'starformer_no_delay' in args.name.lower():
    args.max_delay = None
  train_ds = ds_builder.get_dataset(
    args.batch_size, args.num_workers, random_interval=args.random_interval,
    max_delay=args.max_delay, card_shuffle=args.card_shuffle, use_card_idx=args.pred_card_idx)
  args.n_unit = len(unit_list)
  args.n_cards = ds_builder.n_cards
  args.max_timestep = int(max(ds_builder.data['timestep']))
  args.steps_per_epoch = len(train_ds)
  ### Model ###
  args.no_delay = False
  if 'starformer_3l' in args.name.lower():
    from katacr.policy.offline.starformer import StARConfig, TrainConfig, StARformer
    ModelConfig, Model = StARConfig, StARformer
  if 'vidformer' in args.name.lower():
    from katacr.policy.offline.vidformer import ViDConfig, TrainConfig, ViDformer
    ModelConfig, Model = ViDConfig, ViDformer
  if 'starformer_2l' in args.name.lower():
    from katacr.policy.offline.starformer_2L import StARConfig, TrainConfig, StARformer
    ModelConfig, Model = StARConfig, StARformer
    assert args.pred_card_idx
  if 'starformer_no_delay' in args.name.lower():
    from katacr.policy.offline.starformer_no_delay import StARConfig, TrainConfig, StARformer
    ModelConfig, Model = StARConfig, StARformer
    args.no_delay = True
  if 'DT' in args.name:
    from katacr.policy.offline.dt import DTConfig, TrainConfig, DT
    ModelConfig, Model = DTConfig, DT
  model_cfg = ModelConfig(**vars(args))
  model = Model(model_cfg)
  model.create_fns()
  train_cfg = TrainConfig(**vars(args))
  state = model.get_state(train_cfg, verbose=False)
  ### Checkpoint ###
  ckpt_manager = CheckpointManager(str(args.path_logs / 'ckpt'), max_to_keep=30)
  write_tfboard_freq = min(100, len(train_ds))

  ### Train and Evaluate ###
  for ep in range(args.total_epochs):
    print(f"Epoch: {ep+1}/{args.total_epochs}")
    print("Training...")
    logs.reset()
    bar = tqdm(train_ds, ncols=200)
    for s, a, rtg, timestep, y in bar:
      for x in [s, a, y]:
        for k, v in x.items():
          x[k] = v.numpy()
      rtg = rtg.numpy(); timestep = timestep.numpy()
      B = y['select'].shape[0]
      # a is real action in each frame, y is target action with future action time delay predict
      # we need select=0 and pos=(0,-1) as start action padding idx.
      a['select'] = np.concatenate([np.full((B, 1), 0, np.int32), a['select'][:,:-1]], 1)  # (B, l)
      pad = np.stack([np.full((B, 1), 0, np.int32), np.full((B, 1), -1, np.int32)], -1)
      a['pos'] = np.concatenate([pad, a['pos'][:,:-1]], 1)  # (B, l, 2)
      if args.no_delay:
        state, (loss, (loss_s, loss_p, acc_s, acc_p, acc_su, acc_sp)) = model.model_step(state, s, a, rtg, timestep, y, train=True)
        logs.update(
          ['train_loss', 'train_loss_select', 'train_loss_pos',
          'train_acc_select', 'train_acc_pos', 'train_acc_select_use',
          'train_acc_select_and_pos',],
          [loss, loss_s, loss_p, acc_s, acc_p, acc_su, acc_sp])
        acc_d = loss_d = acc_spd = 0.0
      else:
        state, (loss, (loss_s, loss_p, loss_d, acc_su, acc_p, acc_d, acc_sp, acc_spd)) = model.model_step(state, s, a, rtg, timestep, y, train=True)
        logs.update(
          ['train_loss', 'train_loss_select', 'train_loss_pos', 'train_loss_delay',
          'train_acc_select_use', 'train_acc_pos',
          'train_acc_delay', 'train_acc_select_and_pos',
          'train_acc_select_and_pos_and_delay'],
          [loss, loss_s, loss_p, loss_d, acc_su, acc_p, acc_d, acc_sp, acc_spd])
        acc_s = 0.0
      # print(loss, loss_s, loss_p)
      # print(f"loss={loss:.4f}, loss_select={loss_s:.4f}, loss_pos={loss_p:.4f}, acc_select={acc_s:.4f}, acc_pos={acc_p:.4f}")
      bar.set_description(f"{loss=:.4f}, {loss_s=:.4f}, {loss_p=:.4f}, {loss_d=:.4f}, {acc_s=:.4f}, {acc_su=:.4f}, {acc_p=:.4f}, {acc_d=:.4f}, {acc_sp=:.4f}, {acc_spd=:.4f}")
      if state.step % write_tfboard_freq == 0:
        logs.update(
          ['SPS', 'epoch', 'learning_rate'],
          [write_tfboard_freq / logs.get_time_length(), ep+1, train_cfg.lr_fn(state.step)]
        )
        logs.writer_tensorboard(writer, state.step)
        logs.reset()
    ckpt_manager.save(ep+1, state, vars(args))
  ckpt_manager.close()
  writer.close()
  if args.wandb:
    import wandb
    wandb.finish()

if __name__ == '__main__':
  train()
