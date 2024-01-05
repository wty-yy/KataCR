from katacr.utils.related_pkgs.utility import *
import katacr.detection.cfg as cfg
from katacr.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter, datetime, str2bool

class YOLOv5Args(CVArgs):
  ### Dataset ###
  num_classes: int
  num_data_workers: int
  repeat: int
  ### Augmentation for train ###
  hsv_h: float  # HSV-Hue augmentation
  hsv_s: float  # HSV-Saturation augmentation
  hsv_v: float  # HSV-Value augmentation
  fliplr: float  # flip left-right (probability)
  ### Model ###
  anchors: List[Tuple[int, int]]
  ### Training ###
  accumulate: int  # accumulate the gradient
  use_cosine_decay: bool  # use cosine learning rate decay, else linear decay
  warmup_epochs: int
  steps_per_epoch: int
  learning_rate_final: float
  learning_rate_fn: Callable
  momentum: float
  coef_box: float
  coef_obj: float
  coef_cls: float

def get_args_and_writer(no_writer=False, input_args=None) -> Tuple[YOLOv5Args, SummaryWriter] | YOLOv5Args:
  parser = Parser(model_name="YOLOv5", wandb_project_name=cfg.dataset_name)
  ### Model ###
  parser.add_argument("--anchors", nargs='+', default=cfg.anchors,
    help="the anchors bounding boxes")
  ### Dataset ###
  parser.add_argument("--path-dataset", type=cvt2Path, default=cfg.path_dataset,
    help="the path of the dataset")
  parser.add_argument("--image-shape", nargs='+', default=cfg.image_shape,
    help="the input shape of the YOLOv4 model")
  parser.add_argument("--num-classes", type=int, default=cfg.num_classes,
    help="the number of the classes in dataset")
  parser.add_argument("--hsv-h", type=float, default=cfg.hsv_h,
    help="the fraction of HSV-Hue in image augmentation")
  parser.add_argument("--hsv-s", type=float, default=cfg.hsv_s,
    help="the fraction of HSV-Saturation in image augmentation")
  parser.add_argument("--hsv-v", type=float, default=cfg.hsv_v,
    help="the fraction of HSV-Value in image augmentation")
  parser.add_argument("--fliplr", type=float, default=cfg.fliplr,
    help="the probability of fliping image left and right augmentation")
  parser.add_argument("--num-data-workers", type=int, default=cfg.num_data_workers,
    help="the number of the subprocesses to use for data loading")
  parser.add_argument("--repeat", type=int, default=cfg.repeat,
    help="the repeat times of the training dataset")
  ### Training ###
  parser.add_argument("--total-epochs", type=int, default=cfg.total_epochs,
    help="the total epochs for training")
  parser.add_argument("--batch-size", type=int, default=cfg.batch_size,
    help="the batch size for training")
  parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate_init,
    help="the initial learning rate for training")
  parser.add_argument("--learning-rate-final", type=float, default=cfg.learning_rate_final,
    help="the final learning rate for training")
  parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay,
    help="the coef of the l2 weight penalty")
  parser.add_argument("--warmup-epochs", type=int, default=cfg.warmup_epochs,
    help="the epochs for warming up the learning rate")
  parser.add_argument("--momentum", type=float, default=cfg.momentum,
    help="the momentum for optimizer")
  parser.add_argument("--coef-box", type=float, default=cfg.coef_box,
    help="the coef of the bounding box coordinate loss")
  parser.add_argument("--coef-obj", type=float, default=cfg.coef_obj,
    help="the coef of the object loss")
  parser.add_argument("--coef-cls", type=float, default=cfg.coef_cls,
    help="the coef of the classification loss")
  parser.add_argument("--accumulate", type=str2bool, default=True,
    help="if taggled, accumulate the loss to nominal batch size 64")
  parser.add_argument("--use-cosine-decay", type=str2bool, default=True,
    help="if taggled, cosine learning rate decay will be used, else use the linear learning rate decay")
  args = parser.get_args(input_args)
  args.input_shape = (args.batch_size, *args.image_shape)

  nbc = 32  # nominal batch size
  if args.accumulate:
    args.accumulate = max(round(nbc / args.batch_size), 1)
    args.weight_decay /= args.accumulate
    args.steps_per_epoch = cfg.train_datasize // (args.accumulate * args.batch_size)
  else:
    args.accumulate = 1
    args.steps_per_epoch = cfg.train_datasize // args.batch_size

  args.run_name = (
    f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}"
    f"__batch{'(a)' if args.accumulate > 1 else ''}_{int(args.batch_size*args.accumulate)}"
    f"__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}"
  )
  if no_writer: return args
  
  writer = parser.get_writer(args)
  return args, writer

