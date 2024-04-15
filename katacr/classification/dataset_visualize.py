from train import DatasetBuilder, path_dataset_root, TrainConfig
import cv2

if __name__ == '__main__':
  ds_builder = DatasetBuilder(str(path_dataset_root / "images/card_classification"))
  train_cfg = TrainConfig()
  train_ds = ds_builder.get_dataloader(train_cfg, mode='train')
  val_ds = ds_builder.get_dataloader(train_cfg, mode='val')
  cv2.namedWindow('img', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
  for x, y in train_ds:
    x = x.numpy()
    print(ds_builder.idx2card[int(y[0])])
    cv2.imshow('img', x[0,...,::-1])
    cv2.waitKey(0)
