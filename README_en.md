# Clash Royale Non-Embedded AI

[Chinese](README.md) | English

This repository contains all the code for my undergraduate design project, aiming to implement an agent that makes decisions solely based on information obtained from mobile device screens. The design framework is as follows:

![Framework](asserts/figures/framework_en.png)

Building generative object detection dataset
![Generative object detection dataset](asserts/figures/detection_dataset_building_en.png)

The image dataset used: [GitHub - Clash-Royale-Detection-Dataset Object Recognition and Image Classification Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset).

YOLOv8 Object Detection
<div align="center">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/1.gif?raw=true" width="49%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/2.gif?raw=true" width="49%">
</div>

Policy model struct design
![policy model](asserts/figures/policy_model_en.png)

Offline reinforcement learning strategy with 8000-point AI for real-time battles ([12 winning battles - Bilibili](https://www.bilibili.com/video/BV1xn4y1R7GQ/?vd_source=92e1ce2ebcdd26a23668caedd3c9e57e))
<div align="center">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/1_eval.gif?raw=true" width="100%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/2_eval.gif?raw=true" width="100%">
  <img src="https://github.com/wty-yy/picture-bed/blob/master/3_eval.gif?raw=true" width="100%">
</div>

## Environment Requirements
The inference environment I used: Huawei HarmonyOS for mobile devices, Ubuntu 24.04 LTS for PCs, CPU: R9 7940H, GPU: RTX GeForce 4060 Laptop, average decision time 120ms, feature fusion time 240ms.

At least one Nvidia GPU is required for the PC. Due to the necessity of V4L2 in the Linux kernel for mobile video stream input and the lack of Windows support for the GPU version of `JAX`, the validation and decision-making parts of this project **can only run under Linux**.

## Dependency Installation
The `requirements.txt` lists all the Python packages used in this project. However, since three different neural network frameworks are used, the environment installation is recommended as follows:

1. Install [miniforge](https://github.com/conda-forge/miniforge) and create an environment: `conda create -n katacr python==3.11`

2. Install CUDA according to the highest version supported by your GPU driver (use `nvidia-smi` to check the highest supported CUDA version). It's recommended to install `cuda` directly in the `conda` environment:
```shell
conda activate katacr
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9  # or cudatoolkit=12.0 cudnn=8.9
```

3. Install the neural network frameworks:
   - [Install `JAX`](https://jax.readthedocs.io/en/latest/installation.html) (Note: For CUDA 11.8, use `pip install "jax[cuda11]==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"`, as the latest version no longer supports CUDA 11.8)
   - [Install `PyTorch 2.2.2`](https://pytorch.org/get-started/previous-versions/#v222)
   - [Install `PaddlePaddle 2.6.1`](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)
   - Run `debug/cuda_avail.py` to check if all neural network frameworks support GPU.

4. Install other required packages: `pip install -r requirements.txt`

See [scrcpy/doc/linux](https://github.com/Genymobile/scrcpy/blob/master/doc/linux.md) for Scrcpy installation instructions.

## Model Validation
Different models have different validation methods.

| | Continuous Action Prediction Model (with Delay) | Discrete Action Prediction Model (without Delay) | Continuous Action Prediction Full Card Model |
|-|-|-|-|
| Validation Code | [eval.py](./katacr/policy/offline/eval.py) | [eval_no_delay.py](./katacr/policy/offline/eval_no_delay.py) | [eval_all_unit.py](./katacr/policy/offline/eval_all_unit.py) |
| Model Parameter Download | [StARformer_3L__step50](https://drive.google.com/drive/folders/1kqE_2xDainIixf4u5YD12aqT5_LxiZwZ?usp=drive_link)<br>[DT_4L__step50](https://drive.google.com/drive/folders/1gwkFdxYdjM7gdbiJkcPYRmt2lMXPqZWa?usp=drive_link) | [StARformer_no_delay_2L__step50](https://drive.google.com/drive/folders/1RuS9SgwVOI4C67NVW526F5I1H_KbPXOs?usp=drive_link) | [StARformer_3L_pred_cls__step50](https://drive.google.com/drive/folders/1STbGjjai4gTA8sEbbfZDqmE6M7xweqG_?usp=drive_link) |
| Total Reward (20 test rounds) | −4.7±3.1 <br> −5.7±2.5 | −7.5±0.8 | −5.6±2.1

Place the model weight files into `KataCR/logs/Policy/{model-name}`, and the model validation is performed as follows:
```shell
cd KataCR/katacr/policy/offline
python eval.py --load-epoch 3 --eval-num 20 --model-name "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240512_181646"
python eval.py --load-epoch 8 --eval-num 20 --model-name "DT_4L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240519_224135"
python eval_no_delay.py --load-epoch 1 --eval-num 20 --model-name "StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252"
python eval_all_unit.py --load-epoch 2 --eval-num 20 --model-name "StARformer_3L_pred_cls_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240516_125201"
```

## Model Training
### YOLOv8
See [yolov8_modify](./asserts/yolov8_modify.md) for the reconstruction details of the YOLOv8 model.

- Download the generative dataset [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset) and modify the `path_dataset` parameter in [`build_dataset/constant.py`](./katacr/build_dataset/constant.py) to your local dataset path.

- Generate object recognition images: Run [`build_dataset/generator.py`](./katacr/build_dataset/generator.py), and you can see the generated original images and images with object recognition in the `KataCR/logs/generation` folder.

- YOLOv8 training: Due to the task requiring recognition of up to 150 categories, I attempted YOLOv8 detector combination methods, with each detector recognizing different-sized slices of categories:
![Slice Size Distribution](./asserts/figures/segment_size.jpg)
The orange and green dashed lines divide the types of slices required for double and triple model combinations. The training methods for different model combinations are as follows:
  1. Configure multiple model parameter configuration files [`yolov8/cfg.py`](./katacr/yolov8/cfg.py).
  2. Run [`yolov8/model_setup.py`](./katacr/yolov8/model_setup.py) to automatically generate corresponding detector configurations under `./katacr/yolov8/detector{i}` for model training (required categories, validation set path).
  3. Configure the model name and device GPU number in [`yolov8/ClashRoyale.yaml`](./katacr/yolov8/ClashRoyale.yaml), along with some data augmentation strategies.
  4. Train: Run [`yolov8/train.py`](./katacr/yolov8/train.py) to train the models (training curves for this project can be found at [wandb_YOLOv8](https://wandb.ai/wty-yy/YOLOv8)).
  5. Validate: Use [`yolov8/combo_validator.py`](./katacr/yolov8/combo_validator.py) to validate the combined models.
  6. Inference: Use [`yolov8/combo_detect.py`](./katacr/yolov8/combo_detect.py) to perform inference on the combined models (target tracking algorithm can be specified during inference).

- Decision model training:
  1. Offline dataset creation (can also be downloaded directly from [Clash Royale Replay Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset)):
    1. Divide the original battle videos into episodes using [`build_dataset/cut_episodes.py`](./katacr/build_dataset/cut_episodes.py) based on OCR recognition.
    2. Use [`build_dataset/extract_part.py`](./katacr/build_dataset/extract_part.py) to extract the arena parts from the episodes.
    3. Use [`policy/replay_data/offline_data_builder.py`](./katacr/policy/replay_data/offline_data_builder.py) to fuse features and create offline datasets, with the results saved in the `KataCR/logs/offline/{start-time}/` folder.

  2. Refer to the code in [`train_policy.sh`](./train_policy.sh):
  ```shell
  CUDA_VISIBLE_DEVICES=$1 \  # Specify GPU number (supports only single card training)
  python katacr/policy/offline/train.py --wandb \  # Enable online recording with wandb
    --total-epochs 20 --batch-size 32 --nominal-batch-size 128 \  # Training parameter configuration
   --cnn-mode "cnn_blocks" --name "StARformer_3L_v0.8_golem_ai_interval2" --pred-card-idx --random-interval 2 --n-step 50 \  # Model parameter configuration
    --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai"  # Dataset parameter configuration
  ```
  (Training curves for this project can be found at [wandb_ClashRoyale_Policy](https://wandb.ai/wty-yy/ClashRoyale%20Policy))

## Code Architecture

- `build_dataset/`:
  - Preprocessing of video files (episode segmentation, frame extraction, extraction of different parts of images)
  - Tools for building object recognition datasets (assist in labeling datasets, version management of datasets, generative object recognition, label conversion, and recognition label generation, image slicing extraction)
- `classification/`: Classify hand cards and elixir with ResNet.
- `constants`: Storage for constants (card names and corresponding elixir costs, object recognition category names)
- `detection`: YOLOv5 model reconstructed with JAX (later deprecated)
- `interact`: Real-time interaction testing with mobile phones, including object recognition, text recognition, and GUI.
- `ocr_text`: Includes CRNN reconstructed with JAX (later deprecated) and interface conversion for PaddleOCR.
- `policy`:
  - `env`: Two testing environments:
    - `VideoEnv`: Use video datasets as input, only for debugging whether the model's input corresponds to predictions.
    - `InteractEnv`: Real-time interaction with mobile phones, using multiprocessing to execute perception fusion.
  - `offline`: Contains training and validation for decision models `StARformer` and `DT`, including three CNN test structures `ResNet, CSPDarkNet, CNNBlocks`.
  - `perceptron`: Perception fusion: includes feature generators for `state, action, reward`, and integrates them into `SARBuilder` (perception based on `YOLOv8, PaddleOCR, ResNet Classifier`).
  - `replay_data`: Extract perception features from expert videos, create and test offline datasets.
  - `visualization`: Monitor mobile phone images in real-time, visualize perception fusion features.
- `utils`: Tools for object detection (plotting, coordinate transformation, image data augmentation) and video processing tools related to `ffmpeg`.
- `yolov8`: YOLOv8 source code reconstruction, including data reading, model training, validation, object detection, tracking, model recognition type setting, and parameter configuration.