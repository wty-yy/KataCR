from typing import Callable, Any, Tuple, Sequence, Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import argparse, time
import math
import numpy as np
def cvt2Path(x):
    return Path(x)