# 皇室战争非嵌入式AI

本仓库为我本科设计全部代码，目标是实现一个仅通过移动设备屏幕获取信息并做出决策的智能体，其设计框架如下
![架构](asserts/figures/framework.jpg)

使用到的图像数据集：[目标识别、图像分类数据集](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)，

YOLOv8目标检测
<div align="center">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/1.gif?raw=true" width="49%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/2.gif?raw=true" width="49%">
</div>

<div align="center">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/1_eval.gif?raw=true" width="100%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/2_eval.gif?raw=true" width="100%">
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

  - 执行`debug/cuda_avail.py`查看神经网络框架是否都已支持GPU。

4. 安装其他依赖包：`pip install -r requirements.txt`

## 模型训练

