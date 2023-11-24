# -*- coding: utf-8 -*-
'''
@File  : model_weights.py
@Time  : 2023/11/20 16:25:49
@Author  : wty-yy
@Version : 1.0
@Blog  : https://wty-yy.space/
@Desc  : 
Model weight manager.
'''
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from katacv.utils.parser import CVArgs
def load_weights(
    state: train_state.TrainState, args: CVArgs
  ) -> train_state.TrainState:
  if args.load_id == 0: return state
  path_load = args.path_cp.joinpath(f"{args.model_name}-{args.load_id:04}")
  with open(path_load, 'rb') as file:
    state = flax.serialization.from_bytes(state, file.read())
  print(f"Successfully load weights from '{str(path_load)}'")
  return state

class SaveWeightsManager:
  path_save: Path

  def __init__(self, args: CVArgs, ignore_exist=False, max_to_keep: int = None):
    self.path_cp, self.model_name = args.path_cp, args.model_name
    self.num_save = 1
    self.load_id = args.load_id
    self.max_to_keep = max_to_keep
    self.update_path_save()
    if self.path_save.exists() and not ignore_exist:
      print(f"The weights file '{str(self.path_save)}' already exists, still want to continue? [enter]", end=""); input()
  
  def update_path_save(self):
    self.save_id = self.load_id + self.num_save
    self.path_save = self.path_cp.joinpath(f"{self.model_name}-{self.save_id:04}")
  
  def __call__(self, state: train_state.TrainState):
    self.update_path_save()
    with open(self.path_save, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    print(f"Save weights at '{str(self.path_save)}'")
    if self.max_to_keep and self.num_save > self.max_to_keep:
      delete_id = self.load_id + self.num_save - self.max_to_keep
      path_delete = self.path_cp.joinpath(f"{self.model_name}-{delete_id:04}")
      if path_delete.exists():
        path_delete.unlink()
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
