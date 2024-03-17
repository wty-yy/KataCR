from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.related_pkgs.utility import *
import shutil

from katacr.utils.parser import CVArgs
def load_weights(
    state: train_state.TrainState, args: CVArgs
  ) -> train_state.TrainState:
  if args.load_id == 0: return state
  path_load = args.path_cp.joinpath(f"{args.model_name}-{args.load_id:04}")
  with open(path_load, 'rb') as file:
    state = flax.serialization.from_bytes(state, file.read())
  print(f"Successfully load weights from '{str(path_load)}'")
  return state

def load_weights_orbax(state: train_state.TrainState, path: Path | str):
  weights = ocp.PyTreeCheckpointer().restore(str(path))
  state = state.replace(params=weights['params'], batch_stats=weights['batch_stats'])
  print(f"Successfully load weights from '{str(path)}'")
  return state

class SaveWeightsManager:
  path_save: Path

  def __init__(self, args: CVArgs, ignore_exist=False, max_to_keep: int = None, save_best_key: str = None):
    self.path_cp, self.model_name = args.path_cp, args.model_name
    self.num_save = 1
    self.load_id = args.load_id
    self.max_to_keep = max_to_keep
    self.update_path_save()
    if self.path_save.exists() and not ignore_exist:
      print(f"The weights file '{str(self.path_save)}' already exists, still want to continue? [enter]", end=""); input()
    if save_best_key is not None:
      self.best = {
        'key': save_best_key,
        'val': 0,
        'path_log': args.path_logs_model / "best_weights.log",
        'path_weights': self.path_cp.joinpath(f"{self.model_name}-best")
      }
  
  def update_path_save(self):
    self.save_id = self.load_id + self.num_save
    self.path_save = self.path_cp.joinpath(f"{self.model_name}-{self.save_id:04}")
  
  def __call__(self, state: train_state.TrainState, save_key_val: float = None):
    self.update_path_save()
    print(f"Save weights at '{str(self.path_save)}'")
    with open(self.path_save, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    if self.max_to_keep and self.num_save > self.max_to_keep:
      delete_id = self.load_id + self.num_save - self.max_to_keep
      path_delete = self.path_cp.joinpath(f"{self.model_name}-{delete_id:04}")
      if path_delete.exists():
        path_delete.unlink()
    if self.best['key'] is not None and save_key_val is not None:
      if save_key_val > self.best['val']:
        self.best['val'] = save_key_val
        s = f"Update best weights with {self.best['key']}={save_key_val:.8f}, epoch={self.save_id}"
        print(s)
        shutil.copyfile(str(self.path_save), str(self.best['path_weights']))
        with self.best['path_log'].open('a+') as file:
          file.write(s + '\n')
    self.num_save += 1

if __name__ == '__main__':
  model = nn.Dense(10)
  x = jnp.empty((5, 5))
  variables = model.init(jax.random.PRNGKey(42), x)
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optax.adam(learning_rate=1e-3)
  )

  from types import SimpleNamespace
  args = {
    'path_cp': Path().cwd().joinpath("logs/test_checkpoints"),
    'model_name': 'test',
    'load_id': 10
  }
  args = SimpleNamespace(**args)
  weights_manager = SaveWeightsManager(args, max_to_keep=2)
  for i in range(10):
    weights_manager(state)
