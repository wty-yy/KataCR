from pathlib import Path
from katacr.constants.label_list import unit_list

class_num = len(unit_list)
anchors = [  # input image shape: (896, 576, 3)
  (0.1027, 0.0230), (0.0625, 0.0457), (0.1681, 0.0332),  # 56x36
  (0.0907, 0.0705), (0.2427, 0.0397), (0.1331, 0.0864),  # 28x18
  (0.1674, 0.1168), (0.2210, 0.1586), (0.7768, 0.0759),  # 14x9
]

print(class_num)
