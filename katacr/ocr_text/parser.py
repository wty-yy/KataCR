from katacr.utils.related_pkgs.utility import *
import katacr.ocr_text.constant as const

class OCRArgs():
  ### Model ###
  model_name: str  # OCR_CRNN_BiLSTM or OCR_CRNN_LSTM
  ### Dataset ###
  path_weights: Path
  image_width: Tuple
  image_height: Tuple
  input_shape: Tuple
  class_num: int
  max_label_length: int
  ch2idx: dict; idx2ch: dict

def get_args_and_writer(input_args=None) -> OCRArgs:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", default="OCR_CRNN_BiLSTM",
    help="the name of the model, OCR_CRNN_BiLSTM or OCR_CRNN_LSTM, decided by model weights type")
  parser.add_argument("--path-weights", type=cvt2Path, default=const.path_weights,
    help="the path of the CRNN model weights")
  parser.add_argument("--image-width", type=int, default=const.image_width,
    help="the width of the input image")
  parser.add_argument("--image-height", type=int, default=const.image_height,
    help="the height of the input image")
  parser.add_argument("--max-label-length", type=int, default=const.max_label_length,
    help="the maximum length of the labels")
  parser.add_argument("--character-set", nargs='+', default=const.character_set,
    help="the character set of the dataset")
  args = parser.parse_args(input_args)

  args.character_set = [0] + sorted(ord(c) for c in list(args.character_set))
  args.class_num = len(args.character_set)
  args.ch2idx = {args.character_set[i]: i for i in range(len(args.character_set))}
  args.idx2ch = dict(enumerate(args.character_set))

  args.input_shape = (1, args.image_height, args.image_width, 1)
  return args

