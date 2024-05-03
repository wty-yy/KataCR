import jax, jax.numpy as jnp
from flax.training import train_state
from flax import struct

class TrainState(train_state.TrainState):
  dropout_rng: jax.Array
  grads: dict = struct.field(pytree_node=True)
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

if __name__ == '__main__':
  import optax
  rng = jax.random.PRNGKey(42)
  state = TrainState.create(
    apply_fn=lambda x: x, params={'a': 1}, tx=optax.sgd(1),
    dropout_rng=rng, grads={}, accumulate=4, acc_count=0
  )
