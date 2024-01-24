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

### v0.2 (2024.1.9-2024.1.13)
1. 单位的生成扰动修改为$\mathcal{N}(0,0.2^2)$裁剪到$[-0.5,0.5]$
2. 加入动态调整生成位置的概率分布。
3. 加入相关单位的组合性生成，对small_text,elixir,bar,tower-bar,king-tower-bar的组合生成范围和相关性单位生成进行细分，
   并且这些单位不再自动随机生成，生成概率为`0.2`。
4. 模型取消的对`state[1,...,5]`的预测，只关注类别和从属关系的预测。
5. 加入数据增强：
   1. （蒙板）敌方红色背景布；
   2. （蒙板）单位（非others类）半透明、红色、蓝色、白色、金色。（支持亮度值的随机变换，每种增强的选择概率为`0.05`）
   3. （非others类）水平翻转
6. 加入17种背景板，共25种。
7. 不再考虑对emote的生成与识别
8. 对大型文字的出现位置进行固定，并设置生成概率为`0.01`。
9. 对目标位姿进行重新筛选。
10. 加入新的`king-tower-bar`标签，重新生成对应标签数据集，并修正当前数据集中的错误标签。
**FIX BUG**:
1. 重新定义nms计算，只考虑图层在自己之上的，对`unit_list`，按照图层级别，从高到低求
2. 修改边缘单位保留方法，除text单位外，其余单位都水平平移到图像内。

### v0.3 (2024.1.15-2024.1.16)
对YOLOv5模型进行优化，使其达到官方YOLOv5性能：

100epochs识别训练结果：70.25% AP@50, 55.94% AP@75, 49.18% mAP
1. 修正重大错误：detection模型的输出重整顺序错误
2. 修正模型参数问题，减小1/3模型大小
3. 加入对bias从0.1到0.01的学习率warmup
4. 加入focus初始化
5. 修改weight decay位置，避免梯度计算
6. 加入EMA(Exponential Moving Average)
7. 在NMS前将预测框限制到图像内


### v0.4 (2024.1.17-2024.1.20)
80epochs识别训练结果：73.68% AP@50, 58.65% AP@75, 51.51% mAP
1. 对YOLOv5模型中判断tp（true positive）计算方法进行修正，原来按照0.5IOU阈值取最大置信度的框，导致对于更高IOU阈值的框可能没有被视为tp，导致mAP值偏低。
2. 上调蒙板增强中的金色、白色、蓝色的概率，蓝色亮度随机范围上调。
3. 支持从background中提取切面，加入新版22个tower-bar，加入13个king-tower，15个queen-tower，2个背景版（需要注意，新加入的女武神竞技场各方的横向增加了一行）
4. 加入非单位生成：
   1. 随机在背景中生成，阵亡圣水`blood`、蝴蝶`butterfly`、花朵`flower`、彩条`ribbon`、骷髅头`skull`、金杯`cup`（每个单位有自己的生成范围和生成概率）。
   2. 加入方形和盾形的的单位等级`bar-level`联合生成（和bar同级别生成）
5. 重新加入`emote`生成（包含上下左右四个生成范围）
**FIX BUG**:
1. 修正单位边缘不显示问题。
2. 将tower的NMS筛出比例单独计算，筛出比例设为0.8。

#### v0.4.1(2024.1.21)
1. 将king-tower-bar的生成概率提高到0.8。
2. 将血条组合型生成的概率单独计算，设置为0.9，两种血条的单独生成概率设为0.8, 0.2（在`generator.py`中直接写死了，后续如果有更多子类概率生成，建议优化）

#### v0.4.2(2024.1.22)
1. 在红色背景版中加入红色边界线，解决将边界线识别为`bar1`的情况。
2. 下调king-tower-bar生成概率0.5。

#### v0.4.3(2024.1.22)
80epochs识别训练结果：73.80% AP@50, 58.47 AP@75, 51.43% mAP
1. 上调红色背景版中的红色`alpha=80->100`，显示概率上调到`0.2->0.5`。
2. 平均分配`bar, bar-level`概率（`(0.8,0.2)->(0.5,0.5)`）
3. 将YOLOv5中`coef_obj=1.0->2.0`。
**New tool**:
1. `annotation_helper.py`可辅助标记视频，使用方法：
   1. 先将训练好的YOLO模型放置到`/logs/{model-name}-checkpoints/{model-name}-{load_id}`，例如`YOLOv5_v0.4.3-0080`就放到`logs/YOLOv5_v0.4.3-checkpoints/YOLOv5_v0.4.3-0080`下。
   2. 在命令行解析中加入要识别的视频名称和episode编号，例如：
      ```shell
      python katacr/build_dataset/annotation_helper.py --video-name OYASSU_20210528_episodes --episode 2
      ```
      就会自动对`path_dataset`下`CR/images/part2/{video-name}/{episode}`中的所有未标记图片（没有`json`文件的图片）进行识别，生成和图像名称相对应的`json`文件。
   3. Note: 将视频分段成识别数据集的tool是`/katacr/build_dataset/extract_part.py`

识别模型仍存在的问题：

- 解决了错误地将红色边界线识别为`bar1`的问题。
- 减轻了`bar-level`错误识别的情况，在某些背景版中仍然有错误识别（继续修复）
- `king-tower-bar`仍然无法识别。
- 单位在金色蒙板并“纵向拉伸”后无法识别。