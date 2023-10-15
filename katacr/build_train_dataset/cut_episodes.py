# -*- coding: utf-8 -*-
'''
@File    : split_episodes.py
@Time    : 2023/10/15 20:46:39
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 提取视频中的所有回合，按照帧中出现的特征来判断回合的开始与结束，文字为：
|  Start episode  |  End episode  |
|   card table    |  center word  |
'''
import os, sys
sys.path.append(os.getcwd())
import moviepy.editor as mp
from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils import load_image_array
import constant as const
import cv2

def get_features(path_features: Path) -> Sequence[np.ndarray]:
    features = []
    for path in path_features.iterdir():
        if path.is_file() and path.name[-3:] == 'jpg':
            features.append(load_image_array(path, to_gray=True, keep_dim=False))
    return features

def split_episodes(path_video: Path):
    clip = mp.VideoFileClip(str(path_video))
    fps, duration = clip.fps, clip.duration
    file_name = path_video.name[:-4]
    path_episodes = path_video.parent.joinpath(file_name+"_episodes")
    if path_episodes.exists():
        print(f"The episodes path '{str(path_episodes)} is exists, still continue? [Enter]'"); input()
    path_episodes.mkdir(exist_ok=True)

    start_features = get_features(const.path_features.joinpath("start_episode"))
    end_features = get_features(const.path_features.joinpath("end_episode"))

    episode_frames, episode_num, record = [], 0, False
    bar = tqdm(clip.iter_frames(), total=int(fps*duration))
    for image in bar:
        if image.shape[0] != const.image_size[0]:
            image = cv2.resize(image, const.image_size)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if not record and check_feature_exists(image_gray, start_features):
            episode_num += 1
            episode_frames = []
            record = True
        if record:
            episode_frames.append(image)
        if record and check_feature_exists(image_gray, end_features):
            record = False
            path = path_episodes.joinpath(f"{episode_num}.mp4")
            saved_clip =  mp.ImageSequenceClip(episode_frames, fps=30)
            saved_clip.write_videofile(str(path))
            saved_clip.close()
        bar.set_description(f"Process {episode_num} episode")
    clip.close()

def match_feature(image, feature):
    # assert(image.shape[-1] == feature.shape[-1])
    # print(image.shape, feature.shape)
    result = cv2.matchTemplate(image, feature, cv2.TM_SQDIFF_NORMED)
    return result.min() < const.mse_feature_match_threshold

def check_feature_exists(
        image: np.ndarray,
        features: Sequence[np.ndarray]
    ) -> bool:
    for feature in features:
        if match_feature(image, feature):
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-video", type=cvt2Path, default="/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230917.mp4")
    args = parser.parse_args()
    split_episodes(args.path_video)
    