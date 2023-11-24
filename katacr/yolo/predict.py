from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent.parent))

from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.yolo.yolov4 import logits2cell
from katacr.yolo.yolov4_model import TrainState
from katacr.yolo.build_yolo_target import cell2pixel
from katacr.yolo.metric import logits2prob_from_list, get_pred_bboxes, calc_AP50_AP75_AP, mAP, coco_mAP

@jax.jit
def predict(state: TrainState, images: jax.Array):
  logits = state.apply_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    images, train=False
  )
  pred_cell = [logits2cell(logits[i]) for i in range(3)]
  pred_pixel = [jax.vmap(cell2pixel, in_axes=(0,None,None), out_axes=0)(
    pred_cell[i], 2**(i+3), args.anchors[i]
  ) for i in range(3)
  ]
  pred_pixel_prob = logits2prob_from_list(pred_pixel)
  return pred_pixel_prob

from PIL import Image
def show_bbox(image, bboxes, draw_center_point=False):
  """
  Show the image with bboxes use PIL.

  Args:
    image: Shape=(H,W,3)
    bboxes: Shape=(N,13), last dim means: (x,y,w,h,c,states*7,label)
    draw_center_point: Whether to draw the center point of all the bboxes
  """
  from katacr.utils.detection import plot_box_PIL, build_label2color
  from katacr.constants.label_list import idx2unit
  from katacr.constants.state_list import idx2state
  if type(image) != Image.Image:
    image = Image.fromarray((image*255).astype('uint8'))
  if len(bboxes):
    label2color = build_label2color(range(200))  # same color
  for bbox in bboxes:
    unitid = int(bbox[12])
    text = idx2unit[unitid] + idx2state[int(bbox[5])]
    for i in range(6, 12):
      if bbox[i] != 0:
        text += ' ' + idx2state[int((i-5)*10 + bbox[i])]
    image = plot_box_PIL(image, bbox[:4], text=text, box_color=label2color[unitid], format='yolo', draw_center_point=draw_center_point)
    # print(label, label2name[label], label2color[label])
  image.show()

if __name__ == '__main__':
  from katacr.yolo.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv4 --load-id 75".split())
  args.batch_size = 1
  args.path_cp = Path("/home/wty/Coding/GitHub/KataCR/logs/YOLOv4-checkpoints")

  from katacr.yolo.dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  ds = ds_builder.get_dataset()
  args.steps_per_epoch = len(ds)

  from katacr.yolo.yolov4_model import get_yolov4_state
  state = get_yolov4_state(args)

  from katacr.utils.model_weights import load_weights
  state = load_weights(state, args)

  test_num = 10
  for images, bboxes, num_bboxes in ds:
    images, bboxes, num_bboxes = images.numpy(), bboxes.numpy(), num_bboxes.numpy()
    pred = predict(state, images)

    import numpy as np
    np.set_printoptions(suppress=True)
    pred_bboxes = get_pred_bboxes(pred, conf_threshold=0.1, iou_threshold=0.5)
    for i in range(len(pred_bboxes)):
      show_bbox(images[i], pred_bboxes[i], args.path_dataset.name)
      AP50 = mAP(pred_bboxes[i], bboxes[i][:num_bboxes[i]], iou_threshold=0.5)
      AP75 = mAP(pred_bboxes[i], bboxes[i][:num_bboxes[i]], iou_threshold=0.75)
      AP = coco_mAP(pred_bboxes[i], bboxes[i][:num_bboxes[i]])
      print(f"AP50: {AP50:.2f}, AP75: {AP75:.2f}, AP: {AP:.2f}")
    test_num -= 1
    if test_num == 0:
      break

    # print(calc_AP50_AP75_AP(pred_bboxes, bboxes, num_bboxes))

