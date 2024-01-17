# KataCR
## Abstruct
KataCR is a non-embedded AI for Clash Royale based on RL and CV. Supervised learning to learn policies from videos in YouTube, and reinforcement learning (PPO) from playing with human.

## Update Logs

### v0.1 (2024.1.7)
1. 用SAM扣出的单位，可基于生成地图进行简单随机生成。
2. 基于图层的图像生成，可视化结果输出。
3. 8种背景板，25种单位。
4. 在YOLO上训练，20000example/epoch，训练100epoch，但还是没有收敛，
   并且在验证集上（手工标记数据集）没有性能。

### v0.2 (2024.1.9-2024.1.11)
1. 单位的生成扰动修改为$\mathcal{N}(0,0.2^2)$裁剪到$[-0.5,0.5]$
2. 加入动态调整生成位置的概率分布。
3. 加入相关单位的关联性生成，例如bar和unit，elixir和unit
4. 模型取消的对`state[1,...,5]`的预测，只关注类别和从属关系的预测。
5. 加入数据增强：
   1. （蒙板）敌方红色背景布；
   2. （蒙板）单位（非others类）半透明、红色、蓝色、白色、金色。
   3. 所有单位随机大小缩放。
   4. （非others类）水平翻转
6. 加入17种背景板，共25种。
7. 不再考虑对emote的生成与识别
FIX BUG:
1. 重新定义nms计算，只考虑图层在自己之上的，对`unit_list`，按照图层级别，从高到低求

### v0.3 (2024.1.15-2024.1.16)
对YOLOv5模型进行优化，使其达到官方YOLOv5性能：
1. 修正重大错误：detection模型的输出重整顺序错误
2. 修正模型参数问题，减小1/3模型大小
3. 加入对bias从0.1到0.01的学习率warmup
4. 加入focus初始化
5. 修改weight decay位置，避免梯度计算
6. 加入EMA(Exponential Moving Average)
7. 在NMS前将预测框限制到图像内

### v0.4 (2024.1.17-)
1. 对YOLOv5模型中判断tp（true positive）计算方法进行修正，原来按照0.5IOU阈值取最大置信度的框，导致对于更高IOU阈值的框可能没有被视为tp，导致mAP值偏低。