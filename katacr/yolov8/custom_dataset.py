from ultralytics.data import YOLODataset
from ultralytics.models.yolo import detect

class CustomDataset(YOLODataset):
    def get_labels():
        """customize how you read the mask and numpy object"""


class CustomTrainer(detect.DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return CustomDataset(img_path, mode, batch)

settings = dict(data="", epochs=...)
trainer = CustomTrainer(overrides=settings)
trainer.train()