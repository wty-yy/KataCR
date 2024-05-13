import argparse, time
from tensorboardX.writer import SummaryWriter
from pathlib import Path
from katacr.utils.logs import Logs, MeanMetric
from katacr.build_dataset.constant import path_dataset

def str2bool(x): return x in ['yes', 'y', 'True', '1']
def parse_args_and_writer(input_args=None, with_writer=True) -> tuple[argparse.Namespace, SummaryWriter]:
  parser = argparse.ArgumentParser()
  ### Gobal ###
  parser.add_argument("--name", type=str, default="StARformer_3L")  # or "StARformer_2L" or "StARformer_no_delay" or "ViDformer"
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--wandb", type=str2bool, default=False, const=True, nargs='?')
  ### Training ###
  parser.add_argument("--learning-rate", type=float, default=6e-4)
  parser.add_argument("--total-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--nominal-batch-size", type=int, default=128)
  ### Model ###
  parser.add_argument("--n-embd-global", type=int, default=192)
  parser.add_argument("--n-head-global", type=int, default=8)
  parser.add_argument("--n-embd-local", type=int, default=64)
  parser.add_argument("--n-head-local", type=int, default=4)
  parser.add_argument("--n-block", type=int, default=6)
  parser.add_argument("--n-step", type=int, default=50)
  parser.add_argument("--patch-size", default=(2, 2))
  parser.add_argument("--weight-decay", type=float, default=1e-1)
  parser.add_argument("--cnn-mode", type=str, default="cnn_blocks")  # "csp_darknet" or "resnet"
  parser.add_argument("--use-action-coef", type=float, default=5.0)  # just for "StARformer_no_delay"
  ### Dataset ###
  parser.add_argument("--replay-dataset", type=str, default=str(path_dataset/"replay_data/golem_ai"))
  # parser.add_argument("--replay-dataset", type=str, default=str(path_dataset/"replay_data/golem_ai/WTY_20240419_112947_1_golem_enermy_ai_episodes_1.npy.xz"))
  parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--max-delay", type=int, default=20)
  parser.add_argument("--random-interval", type=int, default=1)
  parser.add_argument("--card-shuffle", type=str2bool, default=True)
  parser.add_argument("--pred-card-idx", type=str2bool, default=False, const=True, nargs='?')

  args = parser.parse_args(input_args)
  assert Path(args.replay_dataset).exists(), "The path of replay buffer must exist"
  args.lr = args.learning_rate
  args.patch_size = [int(x) for x in args.patch_size]
  nbc = args.nominal_batch_size
  args.accumulate = max(round(nbc / args.batch_size), 1)
  args.weight_decay *= args.accumulate * args.batch_size / nbc

  ### Create Path ###
  path_root = Path(__file__).parents[3]
  args.run_name = f"{args.name}_{args.cnn_mode}__nbc{nbc}__ep{args.total_epochs}__step{args.n_step}__{args.seed}__{time.strftime(r'%Y%m%d_%H%M%S')}"
  path_logs = path_root / "logs" / args.run_name
  path_logs.mkdir(parents=True, exist_ok=True)
  args.path_logs = path_logs
  if not with_writer:
    return args

  if args.wandb:
    import wandb
    wandb.init(
      project="ClashRoyale Policy",
      sync_tensorboard=True,
      config=vars(args),
      name=args.run_name,
    )
  writer = SummaryWriter(str(path_logs / "tfboard"))
  return args, writer

logs = Logs(
  init_logs={
    'train_loss': MeanMetric(),
    'train_loss_select': MeanMetric(),
    'train_loss_pos': MeanMetric(),
    'train_loss_delay': MeanMetric(),
    'train_acc_select': MeanMetric(),
    'train_acc_select_use': MeanMetric(),
    'train_acc_pos': MeanMetric(),
    'train_acc_delay': MeanMetric(),
    'train_acc_select_and_pos': MeanMetric(),
    'train_acc_select_and_pos_and_delay': MeanMetric(),
    'SPS': MeanMetric(),
    'epoch': 0,
    'learning_rate': MeanMetric(),
  },
  folder2name={
    'Chart': ['learning_rate', 'SPS', 'epoch'],
    'Metrics': ['train_loss',
                'train_loss_select', 'train_loss_pos', 'train_loss_delay',
                'train_acc_select', 'train_acc_select_use', 'train_acc_pos', 'train_acc_delay',
                'train_acc_select_and_pos', 'train_acc_select_and_pos_and_delay'],
  }
)