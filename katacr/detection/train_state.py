from katacr.utils.related_pkgs.jax_flax_optax_orbax import *

class TrainState(train_state.TrainState):
  batch_stats: dict
  grads: dict
  accumulate: int
  acc_count: int

def zeros_grads(state: TrainState):
  state = state.replace(
    grads=jax.tree_map(lambda x: jnp.zeros_like(x), state.grads)
  )
  return state

def update_grads(state: TrainState):
  state = state.apply_gradients(grads=state.grads)
  state = zeros_grads(state)
  return state

def accumulate_grads(state: TrainState, grads: dict):
  state = state.replace(
    grads=jax.tree_map(lambda x, y: x + y, state.grads, grads),
    acc_count=state.acc_count+1
  )
  state = jax.lax.cond(
    state.acc_count % state.accumulate == 0,
    update_grads,
    lambda state: state,
    state
  )
  return state
