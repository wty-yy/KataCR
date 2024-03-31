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

**NEW TOOL**:
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

#### v0.4.4(2024.1.23, 2024.1.28, 2024.2.1)
80epochs识别训练结果：74.32% AP@50, 60.13 AP@75, 52.94% mAP
1. 血条`bar,bar-level`生成概率上调`0.6->0.9`，并加入单独对`bar`的nms阈值设定，当上层图层对`bar`的覆盖率超过`0.1`则删去（保留尽可能得完整）
2. （没有明显效果，没有加入）生成图像的单位数量`add_unit`上调`30->50`
3. 蒙板亮度调整：金色`(50,80)->(70,80)`，白色`(50,80)->(110,120)`
4. 加入对`elixir,clock`的尺度缩放`scale:(0.5,1.0) prob=1.0`
5. （效果退化，没有加入）单位的垂直缩放`scale:(0.5,0.8) prob=0.1`
6. （效果退化明显，没有加入）优化真实框的设置方法：按照**未被上层图层覆盖**的最大框进行设置（使得边界框更加精准）

**FIX BUG**: 修复`BasePredict`中`compute_tp`的类别从属错误。

**NEW TOOL**：加入`detect.py`实现对视频文件的高效识别，R7-5700x+RTX4080识别速度为`13ms`

#### v0.4.5(2024.2.8)
1. 扩充生成式数据集：23个新的单位（5个法术，2个天空，3个地面；18个部队，4个天空，14个地面）
2. 创建新的图层level划分：`ground_spell`，例如：`poison, rage, tornado`
3. 将雪堆`snow`加入到背景图`background-items`生成当中，生成概率为0.05，生成范围为整个竞技场。
4. 上调`bar,bar-level`概率为`1.0`。
5. YOLO基于当前`segment`中的图像（出去`background`相关的内容）更新锚框`anchors`大小。
6. 生成式数据集可以基于固定的种子进行（注意这样不能中断训练，继续训练需要使用新的种子），并对`path.glob`的结果进行`sorted`，从而对训练结果进行复现。

**NEW TOOL**：加入`build_dataset/dataset_version.py`分别对`annotation`和`segment`中的图像标签数目来管理数据集的当前版本，记录更新内容。

#### v0.4.5.1(2024.2.10)
1. 对`royal-ghost`的透明度蒙板概率上调为`0.5`
2. 当`spell_unit`中有部分出界时，不将其平移到图内（和`text`一样）

**FIX BUG**：修复`x-bow`标签名称错误，将`fliplr`变换放到所有变换之前。

#### v0.4.5.2(2024.2.14~2024.2.15)
1. 基于大矿工心机桶，增加5个新单位，1空中法术，4地面单位（CR Dataset v0.3）
2. 支持对1080x2400（高宽比为h/w=2.22）的屏幕的`part3,4`部分的裁剪，可自己采集视频数据划分`epoch`
3. [做`ffmpeg`命令笔记](https://wty-yy.xyz/posts/50944/)，`fps`转换、视频合并、视频分割

#### v0.4.5.3(2024.2.19~2024.2.21)
1. 基于自己录制的视频`WTY_20240218_episodes_1`，增加11个新的单位，`9`个地面部队，`1`个地面法术，`1`个新的标识`evolution-symbol`

#### v0.4.5.4(2024.2.24~2024.2.26)
1. 基于`WTY_20240222_8spells`，增加3个新法术`earthquake,giant-snowball,mirror`，优化`5`个法术的切片图像。
   注意：生成图像中，所有法术出界均不进行平移，而是进行裁剪。
2. 当前已标记70个单位，还需标记72个单位，将训练epoch上调到80->150。

#### v0.4.5.5(2024.2.27~2024.2.29)  
1. 加入`cannoneer-tower`和`queen-tower`的生成比例（`3:7`）
2. 基于`WTY_20240227_miners`加入9个敌方单位（5个法术，4个地面单位），1个非目标单位`dirt`，1个背景部件`grave`，总计11个。
3. 加入动态采样单位生成。（记录所有单位出现的频次`freq`，按照`1/(freq-freq.min()-1)`对应的分布对单位进行采样，从而使各种单位在训练集中出现的频次基本一致）

#### v0.4.5.6(2024.3.1~2024.3.9)
1. 开源[`Clash Roayle Dataset`](https://github.com/wty-yy/Clash-Royale-Dataset)。
2. 2024.3.2：基于`WTY_20240301`标记242帧，共包含13个新地面单位切片。（还差56个单位）
3. 2024.3.3：基于`WTY_20240303`标记219帧，标记用时165mins完成，筛选切片用时90mins。更新数据集为annotation v0.8, segment v0.10，还差45种切片。
4. 2024.3.4：基于`WTY_20240304_episodes_1`标记266帧，用时159mins完成，筛选切片用时90mins，重新单独计算敌方切片数量（之前将我方和敌方合并起来计算的，有四个切片只有我方有敌方没有），当前还差42种切片，合并起来还差38种切片。
5. 2024.3.5：基于`WTY_20240305_episodes_1`标记320帧，用时124mins完成，筛选切片用时80mins，更新数据集为annotation v0.10, segment v0.10（应该是v0.12但是版本更新代码将v0.9视为了上个版本，导致跳过了两次，已修复），当前敌方单位切片还差35种。
6. 2024.3.6：基于`WTY_20240306_episodes_1`标记344帧，用时160mins完成，筛选切片用时120mins。当前敌方单位切片为还差24种。
7. 22024.3.7：基于`WTY_20240307_episodes_1,2`标记342帧，8地面，4空中部队，2关联性地面法术，用时164mins完成，筛选切片用时126mins完成，数据集版本更新为a0.12,s0.12，当前敌方切片还差12种。
8. 22024.3.8：基于`WTY_20240308_episodes_1,2,3`标记190+54+52帧，6个敌方单位切片，用时82+10+15完成，筛选切片用时70mins完成，数据集版本更新为a0.13,s0.13，当前敌方切片还差6种。
9. 22024.3.9：基于`WTY_20240307_episodes_1~8`标记40+22+36+82+20+52+12+12帧，6个进化单位，以及大量切片的优化，用时27+37+24+30+20+36+4+4mins完成，筛选切片用时150mins完成，数据集版本更新为a0.14,s0.14，完成除去`mirror`之外（`mirror`通过逻辑判断实现）的所有敌方切片制作，总计150种，4431个切片，标记数据集（验证集）大小为6842。

> 待解决问题：
> 1. 数据生成中需要加入新的关联性生成，例如`skeleton-king, skeleton-king-skill`和`tesla-evolution, tesla-evolution-shock`。
> 2. 考虑是否要将`bar`分为两个部分识别`bar-level`和`bar`，还要考虑如何生成`skeleton-king-bar`。

### v0.5 (2024.3.10-2024.3.14)
对Generator进行如下优化：
1. 加入新的`object_unit_list`包含`'axe','dirt','goblin-ball','bomb'`这些物体单位（需要识别，但是不关联生成`bar,bar-level,elixir,clock,small-text`）
2. 更新`drop_units`（不直接生成的单位）：`'skeleton-king-skill', 'tesla-evolution-shock'`
3. 加入新的组件component生成相对点位`center`用于`tesla-evolution,tesla-evolution-shock`和`skeleton-king,skeleton-king-skill`
4. 重新规划组件生成概率计算方法：
   1. 若组件属于`important_components`则单独计算其生成概率。
   2. 若组件不属于`important_components`则先基于`component_prob`的概率$p_s$判断是否生成组件，若生成，则对于要生成的第$i$个组件其对应一个生成概率$p_{c_i}$，当满足概率$p_s\cdot p_{c_i}$使生成该组件。
   基于上述组件生成方法，我们将`elixir,clock,small-text`个数之和的期望设置为$0.2$，之前为$0.2*(1/3+2/3+3/3)=0.4$而且生成较为密集。
5. 在`_build_unit_from_path`中加入生成初始位置`center,left_center,right_center,left_bottom,right_bottom`。
6. 将`bar`和`bar-level`分开进行识别，生成概率仍然为各自$0.5$，当`bar`要生成的时候，随机选一个`bar-level`放置到其左侧并向右微调$2.4 pixel$，并加入整体中心点向左偏移`(-0.3,-0.1) cell`的扰动。
7. 将训练数据集单个epoch从13000上调至100K（和COCO数据集118K对齐），total_epoch需要调试（当前为150）。
8. 基于150种，4431个切片，更新anchor，由于有少量的异常值（`lightning, big-text`），所以将宽度限制在`300`内，高度限制在`400`内。
9. 加入`naive`概率地图变换，将生成过的位置概率直接修改为0。

**FIX BUG**
1. `generator.reset()`时忘记为概率地图的生成进行重置。

### v0.6 (2024.3.23-2024.3.25)
加入YOLOv8对ClashRoyale Dataset的识别，主要重构内容为：加入对从属部队的类别识别（多标签），修改label的读取路径，具体修改内容请见 [YOLOv8 for ClashRoyale Dataset](./asserts/yolov8_modify.md)
1. `v0.6.`: 生成式数据集参数为 `unit_nums=40, map_update_mode='naive', intersect_ratio_thre=0.5, train_datasize=100000`，在21epochs达到最高mAP为68.56%，但是在30epoch时学习发生崩溃，原因并不清楚。
2. `v0.6.2`: 将蒙版概率提高为0.05，每个单位总计有 $0.27$ 概率获得蒙版，修改生成式参数为 `unit_nums=30, map_update_mode='dynamic', intersect_ratio_thre=0.6, train_datasize=50000`，加入附塔的生成概率为0.9，其中queen tower生成概率为0.6，cannoneer tower生成概率为0.3，在52epochs达到最高mAP为66.68%，后基本收敛。
3. `v0.6.3`: 基于 `v0.6.2` 修改生成式参数为 `unit_nums=40, map_update_mode='naive'`，25epochs达到66.78%，后基本收敛。
4. `v0.6.4`：基于 `v0.6.3` 修改 `train_datasize=20000`，在91epochs达到67.9%。
5. `v0.6.5`：基于`v0.6.4`将`map_update_mode='dynamic'`

### v0.7.1~v0.7.3 (2024.3.28-2024.3.30)
使用两个识别器模型为YOLOv8l，分别识别90个类别，最后将重复识别目标用nms处理，具体重构内容请见 [YOLOv8 for ClashRoyale Dataset 双模型识别](./asserts/yolov8_modify.md#双模型识别)。
1. `v0.7.`：每个detector使用参数`unit_nums=40, map_update_mode='naive', intersect_ratio_thre=0.6, train_datasize=20000`。
2. `v0.7.1`：加到3个分类器，分别识别小中大三种不同的目标类型，调整参数`epochs=50`
3. `v0.7.2`：关闭所有蒙版。效果仍然不好
4. `v0.7.3`：蒙版全部开为`0.02`，只有`violet`开为`0.01`，调整参数`unit_nums=30, map_update_mode='dynamic'`

### v0.7.4~v0.7.5 (2024.3.31)
`v0.7.4`: 对 `Generator.py` 加入以下更新内容：
   1. 对从属阵营分别计算`bar,bar-level`的生成概率，我方单位概率为`0.25`，敌方单位概率为`1.0`
   2. 关闭YOLO对text文本的识别（改用PaddleOCR进行文本识别），注：这会导致后续mAP指标下降
   3. 当关联性生成的**主体**消失时，则删去关联性生成的其他部件。
   4. 敌我双方的切片分布不均匀，生成的数量差距可能很大，将动态概率分布的`moveable_unit_path`中的路径，精确到从属的`belong`。
   5. 降低`ribbon`生成数量`50->30`。
   6. 当`bar`超出图片时直接将超出部分进行裁剪。
   7. 对`archer-queen`加入`0.3`的透明概率。
   8. 模型配置修改为`unit_nums=40, intersect_ratio_thre=0.5`
   9. 测试了`yolov8m`，识别效果比`yolov8l`差比较多，不考虑使用。
`v0.7.5`：将分类器个数重新将为两个，每个分类器识别类别数目为`90`，上调`train_datasize=50000`