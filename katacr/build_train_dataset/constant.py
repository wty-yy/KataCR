from pathlib import Path

image_size = (592, 1280)
path_logs = Path.cwd().joinpath("logs")
path_logs.mkdir(exist_ok=True)
path_features = Path.cwd().joinpath("katacr/features")
path_videos = Path("/home/yy/Coding/datasets/CR/fast_pig_2.6")
# assert(path_videos.exists())

split_bbox_params = {
    'part1': {  # number ocr
        'time': (0.835, 0.074, 0.165, 0.025),
        'hp0':  (0.166, 0.180, 0.090, 0.020),
        'hp1':  (0.755, 0.183, 0.090, 0.020),
        'hp2':  (0.515, 0.073, 0.090, 0.020),
        'hp3':  (0.162, 0.617, 0.090, 0.020),
        'hp4':  (0.756, 0.617, 0.090, 0.020),
        'hp5':  (0.511, 0.753, 0.090, 0.020),
    },
    'part2': (0.021, 0.073, 0.960, 0.700),  # battle field
    'part3': (0.000, 0.821, 1.000, 0.179),  # card table
    'part4': {  # center word ocr
        'up': (0.100, 0.340, 0.800, 0.070),
        'mid': (0.180, 0.410, 0.650, 0.050),
    }
}

fps_ocr_train_data = 15
fps_voc_train_data = 3

mse_feature_match_threshold = 0.03
text_features_episode_end = ['match', 'over', 'break']
text_confidence_threshold = 0.005