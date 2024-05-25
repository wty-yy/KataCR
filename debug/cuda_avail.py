from jax.lib import xla_bridge
print("JAX:", xla_bridge.get_backend().platform)

import torch
from torch.utils.cpp_extension import CUDA_HOME
print("Torch:")
print(torch.cuda._is_compiled())
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())

import paddle
print("Paddle:", paddle.device.is_compiled_with_cuda())

"""
Expect outputs:

JAX: gpu
Torch:
True
True
12.1
8907
Paddle: True
"""