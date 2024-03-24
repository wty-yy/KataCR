import argparse, datetime
from pathlib import Path
from tensorboardX import SummaryWriter
from typing import Sequence

def cvt2Path(x): return Path(x)
def str2bool(x): return x in ['yes', 'y', 'True', '1']

"""
Parser提供通用的参数初始化，包括以下所列出的参数
数据集对应的参数需要视情况，在实例化Parser后再进行添加
"""
from typing import NamedTuple
class CVArgs(NamedTuple):
    model_name: str
    wandb_track: bool
    wandb_project_name: str
    path_logs: Path
    path_logs_model: Path
    write_tensorboard_freq: int
    load_id: int
    save_weights_freq: int
    seed: int
    total_epochs: int
    learning_rate: float
    weight_decay: float
    # Related
    path_cp: Path
    run_name: str
    input_shape: tuple
    # Dataset
    path_dataset: Path
    path_dataset_tfrecord: Path
    batch_size: int
    shuffle_size: int
    image_size: int
    image_shape: Sequence[int]
    # Model
    input_shape: int
    train: bool
    evaluate: bool

class Parser(argparse.ArgumentParser):

    def __init__(self, model_name="YoloV1", wandb_project_name="PASCAL VOC"):
        super().__init__()
        # Common
        self.add_argument("--model-name", type=str, default=model_name,
            help="the name of the model")
        self.add_argument("--wandb-track", type=str2bool, default=False, const=True, nargs='?',
            help="if taggled, track with wandb")
        self.add_argument("--wandb-project-name", type=str, default=wandb_project_name)
        self.add_argument("--path-logs", type=cvt2Path, default=Path(__file__).parents[2] / "logs",
            help="the path of the logs")
        self.add_argument("--write-tensorboard-freq", type=int, default=100,
            help="the frequeny of writing the tensorboard")
        # Model weights
        self.add_argument("--load-id", type=int, default=0,
            help="if load the weights, you should pass the id of weights in './logs/{model_name}-checkpoints/{model_name}-{id:04}'")
        self.add_argument("--save-weights-freq", type=int, default=1,
            help="the frequency to save the weights in './logs/{model_name}-checkpoints/{model_name}-{id:04}'")
        # Hyper-parameters
        self.add_argument("--seed", type=int, default=0,
            help="the seed for initalizing the model")
        # model
        self.add_argument("--train", type=str2bool, default=False, const=True, nargs='?',
            help="if taggled, start training the model")
        self.add_argument("--evaluate", type=str2bool, default=False, const=True, nargs='?',
            help="if taggled, start evaluating the model")
    
    @staticmethod
    def check_args(args: CVArgs):
        """
        If you need change `args.model_name` after `parse_args()`, then you need this function
        to build related log directories.
        """
        if args.train and args.evaluate:
            raise Exception("Error: can't both train and evaluate")

        args.path_logs.mkdir(exist_ok=True)
        args.path_cp = args.path_logs.joinpath(args.model_name+"-checkpoints")
        args.path_cp.mkdir(exist_ok=True)
        args.run_name = f"{args.model_name}__load_{args.load_id}__lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
        args.path_logs_model = args.path_logs.joinpath(args.run_name)
        
    def get_args(self, input_args=None) -> CVArgs:
        args = self.parse_args(input_args)
        self.check_args(args)
        return args
    
    @staticmethod
    def get_writer(args: CVArgs=None) -> SummaryWriter:
        if args.wandb_track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=vars(args),
                name=args.run_name,
                save_code=True,
            )
        writer = SummaryWriter(args.path_logs_model)
        writer.add_text(
            "hyper-parameters",
            "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))
        )
        return writer

    
    def get_args_and_writer(self, input_args=None) -> tuple[CVArgs, SummaryWriter]:
        args = self.get_args(input_args)
        return args, self.get_writer(args)