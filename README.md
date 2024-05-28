# 皇室战争非嵌入式AI

中文 | [English](README_en.md)

本仓库为我本科设计全部代码，目标是实现一个仅通过移动设备屏幕获取信息并做出决策的智能体，其设计框架如下
![架构](asserts/figures/framework.jpg)

使用到的图像数据集：[目标识别、图像分类数据集](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)，

YOLOv8目标检测
<div align="center">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/1.gif?raw=true" width="49%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/2.gif?raw=true" width="49%">
</div>

离线强化学习策略与8000分AI进行实时对局
<div align="center">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/1_eval.gif?raw=true" width="100%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/2_eval.gif?raw=true" width="100%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/3_eval.gif?raw=true" width="100%">
</div>

## 环境要求
我使用的推理环境：手机系统为鸿蒙、电脑系统为 Ubuntu24.04 LTS，CPU: R9 7940H，GPU: RTX GeForce 4060 Laptop，平均决策用时 120ms，特征融合用时 240ms。

电脑至少需要一块Nvidia显卡，又由于手机视频流输入必须依赖Linux内核中的`V4L2`，且`JAX`的GPU版本不支持Windows，因此本项目的验证决策部分**只能Linux**系统下运行。

## 依赖包安装
在`requirements.txt`中列举了本项目所用到的Python包，但是由于使用了三种不同的神经网络框架，环境安装建议方法如下：

1. 安装[miniforge](https://github.com/conda-forge/miniforge)，创建环境`conda create -n katacr python==3.11`

2. 根据你的显卡驱动所支持的最高版本安装CUDA（使用`nvidia-smi`查看所支持的最高版本CUDA），建议直接在`conda`环境中安装`cuda`：
```shell
conda activate katacr
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9  # or cudatoolkit=12.0 cudnn=8.9
```

3. 安装神经网络框架：
  - [安装`JAX`](https://jax.readthedocs.io/en/latest/installation.html)（注意：CUDA11.8的编译版本请使用`pip install "jax[cuda11]==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"`，最新版已经不再支持CUDA11.8）
  - [安装`Pytorch 2.2.2`](https://pytorch.org/get-started/previous-versions/#v222)
  - [安装`PaddlePaddle 2.6.1`](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

  - 执行`debug/cuda_avail.py`查看神经网络框架是否都已支持GPU

4. 安装其他依赖包：`pip install -r requirements.txt`

Scrcpy的安装方法见[scrcpy/doc/linux](https://github.com/Genymobile/scrcpy/blob/master/doc/linux.md)

## 模型验证
三种不同模型的验证方法不同
| | 连续动作预测模型（有Delay） | 离散动作预测模型（无Delay） | 连续动作预测全卡牌模型 |
|-|-|-|-|
| 验证代码 | [eval.py](./katacr/policy/offline/eval.py) | [eval_no_delay.py](./katacr/policy/offline/eval_no_delay.py) | [eval_all_unit.py](./katacr/policy/offline/eval_all_unit.py) |
| 模型参数下载 | [StARformer_3L__step50](https://drive.google.com/drive/folders/1kqE_2xDainIixf4u5YD12aqT5_LxiZwZ?usp=drive_link)<br>[DT_4L__step50](https://drive.google.com/drive/folders/1gwkFdxYdjM7gdbiJkcPYRmt2lMXPqZWa?usp=drive_link) | [StARformer_no_delay_2L__step50](https://drive.google.com/drive/folders/1RuS9SgwVOI4C67NVW526F5I1H_KbPXOs?usp=drive_link) | [StARformer_3L_pred_cls__step50](https://drive.google.com/drive/folders/1STbGjjai4gTA8sEbbfZDqmE6M7xweqG_?usp=drive_link) |
| 总奖励（测试20回合） | −4.7±3.1 <br> −5.7±2.5 | −7.5±0.8 | −5.6±2.1

将模型权重文件放到`KataCR/logs/Policy/{model-name}`中，模型验证方法如下：
```shell
cd KataCR/katacr/policy/offline
python eval.py --load-epoch 3 --eval-num 20 --model-name "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240512_181646"
python eval.py --load-epoch 8 --eval-num 20 --model-name "DT_4L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240519_224135"
python eval_no_delay.py --load-epoch 1 --eval-num 20 --model-name "StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252"
python eval_all_unit.py --load-epoch 2 --eval-num 20 --model-name "StARformer_3L_pred_cls_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240516_125201"
```

## 模型训练
### YOLOv8模型
YOLOv8模型的重构内容见[yolov8_modify](./asserts/yolov8_modify.md)。
- 生成式数据集下载[Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)，修改[build_dataset/constant.py](./katacr/build_dataset/constant.py)中`path_dataset`参数为本机的数据集路径。
- 生成式目标识别图像：执行[build_dataset/generator.py](./katacr/build_dataset/generator.py)，即可在`KataCR/logs/generation`文件夹下看到生成的原图像与带目标识别的图像。
- YOLOv8训练：由于本任务识别类别数目多达150个，所以我尝试了YOLOv8检测器组合方法，每个识别器分别对不同切片大小的类别进行识别：
![切片大小分布图](./asserts/figures/segment_size.jpg)
橙色和绿色虚线分别分割了双和三模型组合所需识别的切片类型，训练不同模型组合方法如下：
  1. 配置多模型参数配置文件[`yolov8/cfg.py`](./katacr/yolov8/cfg.py)
  2. 执行[`yolov8/model_setup.py`](./katacr/yolov8/model_setup.py)在`./katacr/yolov8/detector{i}`下生成对应的识别器配置（所需识别的类别，验证集路径）

## 代码架构

- `build_dataset/`：
  - 对视频文件进行预处理（划分episode，逐帧提取，图像不同部分提取）
  - 目标识别数据集搭建工具（辅助标记数据集，数据集版本管理，生成式目标识别，标签转化及识别标签生成，图像切片提取）
- `classification/`：用ResNet进行手牌及圣水分类
- `constants`：常量存储（卡牌名称及对应圣水花费，目标识别类别名称）
- `detection`：自行用JAX复现的YOLOv5模型（后弃用）
- `interact`：测试与手机进行实时交互，包括目标识别，文本识别，GUI
- `ocr_text`：包括用JAX复现的CRNN（后弃用）和PaddleOCR的接口转化
- `policy`：
  - `env`：两种测试环境：
    - `VideoEnv`：将视频数据集作为输入，仅用于调试模型的输入是否与预测相对应
    - `InteractEnv`：与手机进行实时交互，使用多进程方式执行感知融合
  - `offline`：包含了决策模型`StARformer`和`DT`的训练，验证的功能，并包含三种CNN测试结构`ResNet,CSPDarkNet, CNNBlocks`
  - `perceptron`：感知融合：包含了`state,action,reward`三种特征生成器，并整合到`SARBuilder`中（感知基于`YOLOv8,PaddleOCR,ResNet Classifier`）
  - `replay_data`：提取专家视频中的感知特征，制作并测试离线数据集
  - `visualization`：实时监测手机图像，可视化感知融合特征
- `utils`：用于目标检测相关的工具（绘图、坐标转化、图像数据增强），用于视频处理的`ffmpeg`相关工具
- `yolov8`：重构YOLOv8源码，包括数据读取、模型训练、验证、目标检测、跟踪，模型识别类型设置以及参数配置
