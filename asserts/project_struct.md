# 代码架构

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