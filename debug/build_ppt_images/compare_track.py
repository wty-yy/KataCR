import cv2
from subprocess import call
from pathlib import Path

compare_files = [
  [
    "/home/yy/Videos/CR_Detection/v0.7.10/track_musketeer.mp4",
    "/home/yy/Videos/CR_Detection/v0.7.10/notrack_musketeer.mp4",
    (0, 573, 155, 104)
  ],
  [
    "/home/yy/Videos/CR_Detection/v0.7.10/track_littleking.mp4",
    "/home/yy/Videos/CR_Detection/v0.7.10/notrack_littleking.mp4",
    (83, 380, 169, 154)
  ]
]

def build_video():
  for track_file, untrack_file, xywh in compare_files:
    x,y,w,h = xywh
    def process(file):
      name = Path(file).stem
      save_path = Path(file).with_stem(name+'_crop')
      call(["ffmpeg", "-y", "-i", file, "-vf", f"crop={w}:{h}:{x}:{y}", save_path],)
      return str(save_path)
    p1 = process(track_file)
    p2 = process(untrack_file)
    name = Path(track_file).stem.split('_',1)[-1]
    save_path = Path(track_file).with_stem(name+'_merge').with_suffix('.gif')
    call(["ffmpeg", "-y", "-i", p1, "-i", p2, "-filter_complex", "[0:v]pad=iw*2:ih*1[a];[a][1:v]overlay=w", save_path])

if __name__ == '__main__':
  build_video()