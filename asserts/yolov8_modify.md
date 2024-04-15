# YOLOv8 源码修改内容

> `ultralytics.__version__ == 8.1.24`

## 修改数据集集读取路径

由于YOLOv8默认数据集格式为

```shell
- data
	- images
		- 1.jpg
		- 2.jpg
		...
	- labels
		- 1.txt
		- 2.txt
		...
```

而 `ClashRoyale` 数据集更加复杂，在 `imamges` 文件夹下存在按照 `video_name/epoch` 名称划分的子文件夹，所以我的标签直接和图像存放在同一文件夹下，也就是如下格式

```shell
- data
	- images
		- 1.jpg
		- 1.txt
		- 2.jpg
		- 2.txt
		...
```

所以我们先要修改数据集的标签读取路径。观察 `Dataset` 的生成流程：

```python
model.train(...) -> BaseTrainer._do_train -> BaseTrainer._setup_train -> (train_loader, test_loader)
				 -> self.get_dataloader(dataset_path, ...) -> self.build_dataset(dataset_path, ...)
model.val(...) -> BaseValidator.__call__ -> self.get_datasloader(...) -> self.build_dataset(...)
```

重载内容：训练器 `models.yolo.detect.train.DetectionTrainer` 重载为 `CRDetectionTrainer`；验证器 `model.yolo.detect.val.DetectionValidator` 重载为 `CRDetectionValidator`；数据集 `data.YOLODataset` 重载为 `CRDataset`，每个类的重载函数如下：

- 将 `DetectionTrainer.build_dataset, DetectionValidator.build_datset` 返回的数据集换成 `CRDataset` 的实例化结果。
- 重载 `YOLODataset` 中 `get_labels` 函数中标签文件的路径。
- 由于读入新增一个类别，进而需要修改 `cache_label` 中的 `verify_image_label` 函数。

## 修改Model中的Loss

### Model 的生成方法

分析 `model` 的生成路径，`trainer` 中的 `model` 会判断是否重新读取参数 `resume`，如果不重新读取参数，则直接通过 `self.get_model` 创建一个模型，所以我们不仅需要重构 `engine.model.Model` 类的 `task_map` 属性，将每个部件都换成我们自己重构的类（如下代码框所示），还需要重构 `DetectionTrainer.get_model` 方法，同理，当训练时还会用到 `DetectionValidator` 也是通过 `self.get_validator` 创建，所以我们还要重构 `DetectionTrainer.get_validator` 方法。

```python
class YOLO_CR(Model):
  """YOLO (You Only Look Once) object detection model. (Clash Royale)"""

  def __init__(self, model="yolov8n.pt", task=None, verbose=False):
    super().__init__(model=model, task=task, verbose=verbose)

  @property
  def task_map(self):
    """Map head to model, trainer, validator, and predictor classes."""
    return {
      "detect": {
        "model": CRDetectionModel,  # 重载于 nn.task.DetectionModel
        "trainer": CRTrainer,  # 重载于 models.yolo.detect.train.DetectionTrainer
        "validator": CRDetectionValidator,  # 重载于 models.yolo.detect.val.DetectionValidator
        "predictor": CRDetectionPredictor,  # 重载于 engine.predictor.BasePredictor
      },
    }
```

### Loss 的计算方法

`nn.tasks.BaseModel.forward` 会根据模型输入的 `x` 判断调用 `self.loss(x)` 计算损失，或者 `self.predict(x)` 进行预测，如果计算损失则 `nn.task.DetectionModel.loss` 会通过函数 `self.init_criterion` 实例化一个 loss 求解类 `utils.loss.v8DetectionLoss`（不同任务的模型仅需要重载该函数即可）从而创建一个 `self.criterion`，所以我们需要重载 `v8DetectionLoss` 类，称之为 `CRDetectionLoss`，重载内容如下：

- `DetectionModel.init_criterion` 中实例化的 loss 求解器换成 `CRDetectionLoss`。
- `v8DetectionLoss` 中重载的函数为 `__init__, preprocess, __call__`，由于其还进一步实例化任务对齐类 `TaskAlignedAssigner` 为 `self.assigner` 对计算 Loss 所需的标签进行对齐，我们将 `TaskAlignedAssigner` 重载为 `CRTaskAlignedAssigner`。
- 重载 `TaskAlignedAssigner` 中的 `get_target, get_box_metrics` 函数。

## 训练中的图像显示

在读取完 `Dataset` 并创建完 `DataLoader` 后（`engine.trainer.BaseTrain._setup_train` 中完成），进入 `BaseTrain._do_train` 中，当开启 `plot` 选项后，会调用 `self.plot_training_samples` 对当前 `batch` 中的图像进行打印输出，所以为了方便调试，我们也需要重载该函数，同理，在 `models.yolo.detect.val.DetectionValidator` 中同样有 `plot_val_samples` 和 `plot_predictions` 函数会对验证数据集进行打印调试（虽然不清楚为什么不在每个epoch结束时，候进行验证时绘制图像），重载内容如下：

- `DetectionTrainer.plot_training_sample` 中调用函数 `utils.plotting.plot_image` 需要重载。
- `DetectionValidator.plot_val_samples` 和 `DetectionValidator.plot_predictions` 中同样还是函数 `utils.plotting.plot_image` 需要重载，可以和上面重载为同一函数。
- 由于我们的训练数据集是自动生成的，不会将标签全部预读入，所以无法分析图像标签分布，我们需要将 `DetectionTrainer.plot_training_labels` 直接 `pass` 掉。

## 修改验证集的 AP 计算

重构 `DetectionValidator` 中的如下内容：

- `DetectionValidator.postprocess` 中的 `utils.ops.non_max_suppression` 函数（非最大值抑制）。
- `DetectionValidator.update_metrics` 中的 `self._prepare_batch` 函数。

## 修改 Train Dataset 为生成式数据集

我们之前对 `YOLODataset`重载为了 `CRDataset`，对其进行进一步重载，根据 `self.img_path` 是否存在来判断是否使用生成式数据集：

- 当 `__init__` 中 `kwargs['img_path'] == None` 时，使用生成式数据集，重载 `self.__getitem__` 函数，需要额外注意的是，`YOLOv8` 模型要求输入的图像长宽必须为 `32` 的倍数，生成数据集默认输出为 `(568, 896)`，最近的 `32` 倍数为 `(896, 576)`，所以需要对图像和标记框进行等比例缩放以及填充。`__getitem__` 返回结果是一个 `dict`，包含如下必要关键字：
  - `img`：`shape=(3, H, W)` RGB 图像，不需要归一化，需要将长宽填充为 `32` 的倍数。
  - `cls`：`shape=(n, 2)` 分别表示目标框的类别与从属阵营。
  - `bboxes`：`shape=(n, 4)` 目标框格式为 `cxcywh` 也就是 `YOLO` 格式，需要按照图像长宽进行归一化。
  - `batch_idx`：`torch.zeros(n)` 用于在 `DataLoader` 的 `collect` 函数中对一个样本下的标签进行标记（初始全部为 `0`）
  - `im_file`：`None`，图像的路径（用于训练中 `plot_images` 调试图像，如果为 `None` 则不在图像左上角显示其从属的路径位置）
- 当 `__init__` 中 `kwargs['img_path'] != None` 时，说明为常规数据集，直接调用父类函数。
- 相应的需要对 `__len__` 返回的数据集大小进行重构，可以自定义生成式数据集的大小，我定义的为 `100000`（和COCO数据集 `118287` 大小对齐）。

## 预测标签

需要重载如下内容：

```python
predict -> DetectionPredictor -> self.postprocess -> engine.result.Result
		-> self.boxes -> Boxes -> cls, conf
    	-> self.plot(...), self.verbose(...), self.save_txt(...), self.save_crop(...), self.tojson(...)
```

## 双模型识别
使用方法先通过 `yolov8/model_setup.py` 生成模型的配置文件（需要识别的类别等信息），再配置模型训练参数（`batch_size, devices` 等），最后启动训练 `yolov8/train.py --detector 1`（启动第一个 `detector` 的训练）。
### 模型配置文件
通过两个 `YOLOv8l` 做两个识别模型，为每个模型做一个 `id->label` 的配置文件，保存在 `katacr/yolov8/detector{1, 2}/data.yaml` 文件下，通过 `katacr/yolov8/model_setup.py` 生成对应的配置文件。

### 解决共享标签文件的问题
由于两个 `detector` 的验证集需要共享同一个标签文件，但标签文件的类别是按照全部类别进行标记的，所以我们需要筛选出需要识别的目标框做验证，同理，对 `build_dataset/generator.py` 中还需加入对生成单位的种类的控制。

YOLOv8重构内容如下：
1. `YOLODataset.__getitem__` 和标签读取函数 `verify_image_label`。

## 目标追踪
重构`on_predict_start`和`on_predict_postprocess_end`，重载内容：
1. `trackers.`