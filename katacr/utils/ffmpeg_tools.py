import os, proglog
from subprocess import DEVNULL
import subprocess as sp
from moviepy.config import get_setting

# Refer: moviepy.tools.subprocess_call
def subprocess_call(cmd, logger='bar', errorprint=True):
  """ Executes the given subprocess command.
  
  Set logger to None or a custom Proglog logger to avoid printings.
  """
  logger = proglog.default_bar_logger(logger)
  logger(message='Moviepy - Running:\n>>> '+ " ".join(cmd))

  popen_params = {"stdout": DEVNULL,
          "stderr": sp.PIPE,
          "stdin": DEVNULL}

  proc = sp.Popen(cmd, **popen_params)

  out, err = proc.communicate() # proc.wait()
  proc.stderr.close()

  if proc.returncode:
    if errorprint:
      logger(message='Moviepy - Command returned an error')
    raise IOError(err.decode('utf8'))
  else:
    logger(message='Moviepy - Command successful')

  del proc

# Refer: moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip
def ffmpeg_extract_subclip(filename, t1, t2, targetname=None, no_audio=True):
  """ Makes a new video file playing video file ``filename`` between
    the times ``t1`` and ``t2``. """
  name, ext = os.path.splitext(filename)
  if targetname is None:
    T1, T2 = [int(1000*t) for t in [t1, t2]]
    targetname = "%sSUB%d_%d%s" % (name, T1, T2, ext)
  
  cmd = [get_setting("FFMPEG_BINARY"),
    "-y", # cover same file
    "-ss", "%0.2f"%t1,  # start time
    "-i", filename,  # input file
    "-t", "%0.2f"%(t2-t1),  # 
    "-c:v", "libx264",  # codec use h264
    "-an" if no_audio else "",  # no audio (default use audio)
    targetname]
  
  subprocess_call(cmd)
 
if __name__ == '__main__':
  # Test
  ffmpeg_extract_subclip("/home/yy/Coding/datasets/CR/videos/WTY_20240213.mp4", 3.3, 13.3)
