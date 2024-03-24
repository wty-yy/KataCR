from katacr.utils.related_pkgs.utility import *
import katacr.detection.cfg as cfg
from katacr.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter, datetime, str2bool

class YOLOv5Args(CVArgs):
  ### Dataset ###
  num_classes: int
  num_data_workers: int
  train_datasize: int
  num_unit: int
  intersect_ratio_thre: int
  generation_map_mode: str
  ### Augmentation for train ###
  # hsv_h: float  # HSV-Hue augmentation
  # hsv_s: float  # HSV-Saturation augmentation
  # hsv_v: float  # HSV-Value augmentation
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
  learning_rate_bias_fn: Callable
  momentum: float
  coef_box: float
  coef_obj: float
  coef_cls: float

def get_args_and_writer(no_writer=False, input_args=None) -> Tuple[YOLOv5Args, SummaryWriter] | YOLOv5Args:
  parser = Parser(model_name="YOLOv5_v0.5", wandb_project_name=cfg.dataset_name)
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
  # parser.add_argument("--hsv-h", type=float, default=cfg.hsv_h,
  #   help="the fraction of HSV-Hue in image augmentation")
  # parser.add_argument("--hsv-s", type=float, default=cfg.hsv_s,
  #   help="the fraction of HSV-Saturation in image augmentation")
  # parser.add_argument("--hsv-v", type=float, default=cfg.hsv_v,
  #   help="the fraction of HSV-Value in image augmentation")
  parser.add_argument("--fliplr", type=float, default=cfg.fliplr,
    help="the probability of fliping image left and right augmentation")
  parser.add_argument("--num-data-workers", type=int, default=cfg.num_data_workers,
    help="the number of the subprocesses to use for data loading")
  parser.add_argument("--train-datasize", type=int, default=cfg.train_datasize,
    help="the size of training dataset")
  parser.add_argument("--num-unit", type=int, default=cfg.num_unit,
    help="the number of units in images (dataset)")
  parser.add_argument("--intersect-ratio-thre", type=int, default=cfg.intersect_ratio_thre,
    help="the threshold of intersection ratio (dataset)")
  parser.add_argument("--generation-map-mode", type=str, default=cfg.generation_map_mode,
    help="the mode of updating the probability map")
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
  parser.add_argument("--use-cosine-decay", type=str2bool, default=False,
    help="if taggled, cosine learning rate decay will be used, else use the linear learning rate decay")
  # args = parser.get_args(input_args)
  args = parser.parse_args(input_args)
  args.model_name = args.model_name + f"_s{args.seed}"
  parser.check_args(args)  # make directory
  args.input_shape = (args.batch_size, *args.image_shape)
  args.image_size = args.image_shape[:2][::-1]

  nbc = 64  # nominal batch size
  args.accumulate = max(round(nbc / args.batch_size), 1)
  args.weight_decay *= args.accumulate * args.batch_size / nbc
  args.steps_per_epoch = args.train_datasize // (args.accumulate * args.batch_size)

  args.run_name = (
    f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}"
    f"__batch{'(a)' if args.accumulate > 1 else ''}_{int(args.batch_size*args.accumulate)}"
    f"__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}"
  )
  if no_writer: return args
  
  args.path_logs_model.mkdir(exist_ok=True)
  writer = parser.get_writer(args)
  return args, writer

