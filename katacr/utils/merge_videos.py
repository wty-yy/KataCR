import subprocess, os
from pathlib import Path

def get_duration(path_file):
  cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split()
  ret = subprocess.check_output(cmd + [path_file])
  ret = float(ret.decode().strip())
  return ret

def merge_videos_left_and_right(path_file1, path_file2):
  durations = [get_duration(path_file1), get_duration(path_file2)]
  swap = False
  path_file_merge = Path(path_file1).parent / f"merge_{Path(path_file1).stem}_with_{Path(path_file2).name}"
  if durations[0] < durations[1]:
    swap = True
    path_file1, path_file2 = path_file2, path_file1
    durations = durations[::-1]
  path_file2_scale = Path(path_file2)
  path_file2_scale = path_file2_scale.with_stem(path_file2_scale.stem + "_scale")
  cmd = f"ffmpeg -y -i {path_file2} -filter_complex 'setpts=PTS/({durations[1]}/{durations[0]})' {path_file2_scale}"
  print("CMD:", cmd)
  os.system(cmd)
  # subprocess.run(cmd)
  if swap:
    path_file1, path_file2_scale = path_file2_scale, path_file1
  cmd = f"ffmpeg -y -i {path_file1} -i {path_file2_scale} -filter_complex hstack {path_file_merge}".split()
  subprocess.run(cmd)
  path_file2_scale.unlink()

if __name__ == '__main__':
  print(get_duration('/home/yy/Coding/GitHub/KataCR/logs/intercation/20240507_13:58:28/1_predict.mp4'))
  file1 = "/home/yy/Coding/GitHub/KataCR/logs/intercation/20240507_13:58:28/1.mp4"
  file2 = "/home/yy/Coding/GitHub/KataCR/logs/intercation/20240507_13:58:28/1_predict.mp4"
  merge_videos_left_and_right(file1, file2)
