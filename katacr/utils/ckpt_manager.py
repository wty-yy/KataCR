import orbax.checkpoint as ocp
from pathlib import Path
import shutil
from flax.training import train_state

class CheckpointManager(ocp.CheckpointManager):
  def __init__(self, path_save, max_to_keep=1, remove_old=False):
    self.path_save = path_save = str(Path(path_save).resolve())
    if remove_old:
      shutil.rmtree(path_save, ignore_errors=True)
    super().__init__(
      path_save,
      options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, step_format_fixed_length=3),
      item_names={'params', 'config'},
      item_handlers={'params': ocp.StandardCheckpointHandler(), 'config': ocp.JsonCheckpointHandler()}
    )
  
  def save(self, epoch: int, state: train_state.TrainState, config: dict, verbose: bool = True):
    args = ocp.args
    for k, v in config.items():
      if isinstance(v, Path): config[k] = str(v)
    config['_step'] = int(state.step)
    if verbose:
      print(f"Save weights at {self.path_save}/{epoch:03}/")
    return super().save(epoch, args=args.Composite(
      params=args.StandardSave(state.params),
      config=args.JsonSave(config)
    ))
  
  def restore(self, epoch: int):
    return super().restore(epoch)
  
  def load(self, state: train_state.TrainState, epoch: int, need_opt: bool = False):
    ret = self.restore(epoch)
    state = state.replace(step=ret['config']['_step'], params=ret['params'])
    return state
