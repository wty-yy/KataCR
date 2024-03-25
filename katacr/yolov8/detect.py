from katacr.yolov8.train import YOLO_CR

if __name__ == '__main__':
  # model = YOLO_CR("/home/yy/Coding/GitHub/KataCR/logs/yolov8_overfit_train/overfit_train_best_20240324_1545.pt")
  model = YOLO_CR("/home/yy/Coding/GitHub/KataCR/runs/detect/unit40_naive_1e5/best.pt")
  model.predict("/home/yy/Coding/GitHub/KataCR/logs/split_video/OYASSU_20230203_episodes_2.mp4", save=True)