import os, shutil
from pathlib import Path

def convert_video(input_path, output_path):
  cmd = f"ffmpeg -i {input_path} {output_path}"
  print("Execute command:", cmd)
  os.system(cmd)

def compress_video(vid_path):
  vid_path = Path(vid_path)
  stem = vid_path.stem
  vid_small_path = vid_path.with_stem(stem+"_small")
  convert_video(vid_path, vid_small_path)
  shutil.move(vid_small_path, vid_path)
  print("Compress video:", vid_path)

if __name__ == '__main__':
  convert_video("/home/yy/Coding/GitHub/KataCR/logs/interaction/20240528_12:37:56/1.mp4", "/home/yy/Coding/GitHub/KataCR/logs/interaction/20240528_12:37:56/1_new.mp4")

