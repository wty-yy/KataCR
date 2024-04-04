"""
Import jax usefull packages conveniently.

from katacv.utils.related_pkgs.jax import *  # jax, jnp, flax, nn, train_state, optax
"""

import os
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # allocate GPU memory as needed
# os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint as ocp

from functools import partial