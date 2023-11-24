from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.detection import iou

@partial(jax.jit, static_argnames=['iou_ignore'])
def build_target(
    params: jax.Array,  # (M, 5), (x,y,w,h,label)
    num_bboxes: int,  # w, 
    # Shape: [(3, Hi, Wi, 4) for i in range(3)]
    logits: List[jax.Array],
    anchors: jax.Array,  # (3, 3, 2)
    iou_ignore: float = 0.5,
    anchors_threshold: float = 4.0
  ) -> Tuple[List[jax.Array], List[jax.Array]]:
  """
  Return: Tuple -> (target, mask_noobj)
    target[i].shape = (3, Hi, Wi, 13)  # (x,y,w,h,c,states*7,label)
    mask_noobj[i].shape = (3, Hi, Wi)
    i = 0, 1, 2 means three diff scales 2 ** (i + 3)
  """
  params = jnp.array(params)
  target = [jnp.zeros((*logits[i].shape[:3],13)) for i in range(3)]
  mask = [jnp.zeros(logits[i].shape[:3], dtype=jnp.bool_) for i in range(3)]
  bboxes, labels = params[:, :4], params[:, 4:]
  i = 0
  def loop_bbox_i_fn(i, value):
    target, mask = value
    bbox, label = bboxes[i], labels[i]
    # Update ignore examples
    for j in range(3):
      iou_pred = iou(
        bbox, logits[j][...,:4].reshape(-1,4)
      ).reshape(mask[j].shape)
      # label don't need correct, because the confidence is Pr(obj)
      # the last classification predict is Pr(class=C|obj)
      mask[j] = mask[j] | (iou_pred > iou_ignore)
    # Update all anchor target which width and hegiht ratio between (1/4, 4)
    rate = bbox[None,None,2:4] / anchors  # wh ratio
    flag = jnp.maximum(rate, 1.0 / rate).max(-1) < anchors_threshold

    def update_fn(value):
      target, mask, k, b, cell = value
      target = target.at[k,cell[1],cell[0]].set(jnp.r_[b, 1, label])
      mask = mask.at[k,cell[1],cell[0]].set(True)
      return target, mask

    for j in range(3):
      scale = 2 ** (j+3)
      cell = (bbox[:2]/scale).astype(jnp.int32)
      b = jnp.concatenate([bbox[:2]/scale - cell.astype(jnp.float32), bbox[2:4]])
      for k in range(3):
        target[j], mask[j] = jax.lax.cond(
          flag[j,k], update_fn, lambda x: x[:2], (target[j], mask[j], k, b, cell)
        )
    return target, mask
  target, mask = jax.lax.fori_loop(0, num_bboxes, loop_bbox_i_fn, (target, mask))
  for i in range(3):
    mask[i] = (mask[i] ^ True)[..., None]
  return target, mask

@jax.jit
def cell2pixel_coord(xy: jax.Array, scale: int):
  assert(xy.shape[-1] == 2)
  origin_shape, W, H = xy.shape, xy.shape[-2], xy.shape[-3]
  if xy.ndim == 3: xy = xy.reshape(-1, H, W, 2)
  dx, dy = [jnp.repeat(x[None,...], xy.shape[0], 0) for x in jnp.meshgrid(jnp.arange(W), jnp.arange(H))]
  return jnp.stack([(xy[...,0]+dx)*scale, (xy[...,1]+dy)*scale], -1).reshape(origin_shape)

@jax.jit
def cell2pixel(
  output: jax.Array,  # Shape: (3,Hi,Wi,5+num_classes)
  scale=int,  # 8 or 16 or 32
  anchors=jax.Array  # Shape: (3,2)
):
  xy = cell2pixel_coord(output[...,:2], scale)
  def loop_fn(carry, x):
    output, anchor = x
    return None, (output[...,2:3] * anchor[0], output[...,3:4] * anchor[1])
  _, (w, h) = jax.lax.scan(loop_fn, None, (output, anchors))
  return jnp.concatenate([xy, w, h, output[...,4:]], axis=-1)

from PIL import Image, ImageDraw
def plot_rectangle_PIL(image, xyxy, fill=(255,255,255)):
  if type(image) != Image.Image:
    image = Image.fromarray((image*255).astype('uint8'))
  draw = ImageDraw.Draw(image)
  draw.rectangle(xyxy, fill)
  return image

import numpy as np
def test_target_show(image, target, mask, anchors):
  result_bboxes = []
  fill_colors = [(255,255,255), (0,255,0), (0,0,255)]
  for i in range(2,-1,-1):
    cvt_target = cell2pixel(target[i], scale=2**(i+3), anchors=jnp.ones_like(anchors[i]))
    scale = 2**(i+3)
    for j in range(3):

      idxs = np.transpose((1-mask[i][j]).nonzero())
      for idx in idxs:
        x1 = idx[1] * scale; y1 = idx[0] * scale
        image = plot_rectangle_PIL(image, (x1, y1, x1 + scale, y1 + scale), fill_colors[i])
      # print(f"Scale {i}, anchor {j}, target index:", idxs)
      # print("Anchor:", anchors[i,j])

      params = cvt_target[j][cvt_target[j,...,4]==1]  # (N,5+num_classes)
      for x,y,w,h,c,*label in params:
        result_bboxes.append(np.stack([x,y,w,h,*label]))
  if len(result_bboxes):
    result_bboxes = np.stack(result_bboxes)
  print("num bbox:", len(result_bboxes))
  print("mask postive num:", sum([(1-mask[i]).sum() for i in range(3)]))
  from katacr.yolo.dataset import show_bbox
  show_bbox(image, result_bboxes, draw_center_point=True)

if __name__ == '__main__':
  from katacr.yolo.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True)
  args.batch_size = 2
  from katacr.yolo.dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  ds = ds_builder.get_dataset()
  bar = tqdm(ds)
  max_relative_w, max_relative_h = 0, 0
  ws, hs = [], []
  for images, params, num_bboxes in bar:
    images, params, num_bboxes = images.numpy(), params.numpy(), num_bboxes.numpy()
    logits = [jnp.empty((
        args.batch_size,
        3, args.image_shape[0]//(2**i),
        args.image_shape[0]//(2**i),
        12
      )) for i in range(3, 6)
    ]
    print("total box num:", num_bboxes[0])
    print(params[0][:num_bboxes[0]])
    # target, mask = build_target(params[0], num_bboxes[0], [x[0] for x in pred], args.anchors)
    target, mask = jax.vmap(build_target, in_axes=(0,0,0,None), out_axes=0)(
      params, num_bboxes, logits, args.anchors
    )
    for i in range(3):
      obj_target = target[i][target[i][...,4]==1]
      ws.append(obj_target[:,2])
      hs.append(obj_target[:,3])
      max_relative_w = max(max_relative_w, jnp.max(target[i][..., 2]))
      max_relative_h = max(max_relative_h, jnp.max(target[i][..., 3]))
      # print(target[i].shape, mask[i].shape)
    bar.set_description(f"max w: {max_relative_w:.2f}, max h: {max_relative_h:.2f}")

    # Target show
    for i in range(args.batch_size):
      test_target_show(images[i], [x[i] for x in target], [x[i] for x in mask], args.anchors)
    break

  # with open("./logs/target_wh.npy", 'wb') as file:
  #   ws = np.concatenate(ws, axis=0)
  #   hs = np.concatenate(hs, axis=0)
  #   np.save(file, {'ws': ws, 'hs': hs}, allow_pickle=True)
  # print(max_relative_w, max_relative_h)
