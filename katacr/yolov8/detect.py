from katacr.yolov8.train import YOLO_CR

if __name__ == '__main__':
  # model = YOLO_CR("/home/yy/Coding/GitHub/KataCR/logs/yolov8_overfit_train/overfit_train_best_20240324_1545.pt")
  model = YOLO_CR("/home/yy/Coding/GitHub/KataCR/runs/detector1_v0.7.8.pt")
  # model = YOLO_CR("/home/yy/Coding/GitHub/KataCR/runs/detector3_v0.7.6_last.pt")
  # model.predict("/home/yy/Coding/GitHub/KataCR/logs/split_video/OYASSU_20230203_episodes_2.mp4", save=False, show=True)
  # model.predict("/home/yy/Coding/GitHub/KataCR/logs/split_video/detection_test_small_units_test_30fps.mp4", save=False, show=True, conf=0.25)
  model.track("/home/yy/Coding/GitHub/KataCR/logs/split_video/detection_test_small_units_test_30fps.mp4", persist=True, save=False, show=True, tracker="./katacr/yolov8/bytetrack.yaml")
