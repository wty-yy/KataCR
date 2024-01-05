import sys, os
sys.path.append(os.getcwd())
from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
import numpy as np

if __name__ == '__main__':
  ### Initialize arguments and tensorboard writer ###
  from katacr.detection.parser import get_args_and_writer
  args, writer = get_args_and_writer()
  
  ### Initialize log manager ###
  from katacr.detection.logs import logs

  ### Initialize model state ###
  from katacr.detection.model import get_state
  state = get_state(args, use_init=not args.load_id)

  ### Load weights ###
  from katacr.utils.model_weights import load_weights
  if args.load_id > 0:
    state = load_weights(state, args)

  ### Save config ###
  from katacr.utils.model_weights import SaveWeightsManager
  save_weight = SaveWeightsManager(args, ignore_exist=True, max_to_keep=5)
  
  from katacr.detection.dataset_builder import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  train_ds = ds_builder.get_dataset(subset='train')
  val_ds = ds_builder.get_dataset(subset='val')

  ### Build predictor for validation ###
  from katacr.detection.predict import Predictor
  predictor = Predictor(args, state)

  ### Build loss updater for training ###
  from katacr.detection.loss import ComputeLoss
  compute_loss = ComputeLoss(args)

  ### Train and evaluate ###
  start_time, global_step = time.time(), 0
  if args.train:
    for epoch in range(args.load_id+1, args.total_epochs+1):
      print(f"epoch: {epoch}/{args.total_epochs}")
      print("training...")
      logs.reset()
      bar = tqdm(train_ds)
      for x, tbox, tnum in bar:
        x, tbox, tnum = x.numpy().astype(np.float32) / 255.0, tbox.numpy(), tnum.numpy()
        global_step += 1
        state, metrics = compute_loss.step(state, x, tbox, tnum, train=True)
        logs.update(
          [
            'loss_train', 'loss_box_train', 'loss_obj_train', 'loss_cls_train',
          ],
          metrics
        )
        bar.set_description(f"loss={metrics[0]:.4f}, lr={args.learning_rate_fn(state.step):.8f}")
        if global_step % args.write_tensorboard_freq == 0:
          logs.update(
            ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
            [
              args.write_tensorboard_freq/logs.get_time_length(),
              global_step/(time.time()-start_time),
              epoch,
              args.learning_rate_fn(state.step),
            ]
          )
          logs.writer_tensorboard(writer, global_step)
          logs.reset()
      print("validating...")
      logs.reset()
      predictor.reset(state=state)
      for x, tbox, tnum in tqdm(val_ds):
        x, tbox, tnum = x.numpy().astype(np.float32) / 255.0, tbox.numpy(), tnum.numpy()
        predictor.update(x, tbox, tnum)
        _, metrics = compute_loss.step(state, x, tbox, tnum, train=False)
        logs.update(
          ['loss_val', 'loss_box_val', 'loss_obj_val', 'loss_cls_val'],
          metrics
        )
      p50, r50, ap50, ap75, map = predictor.p_r_ap50_ap75_map()
      for name, val in zip(['P@50_val', 'R@50_val', 'AP@50_val', 'AP@75_val', 'mAP_val'], [p50, r50, ap50, ap75, map]):
        print(f"{name}={val:.4f}", end=' ')
      print()
      logs.update(
        [
          'P@50_val', 'R@50_val', 'AP@50_val', 'AP@75_val', 'mAP_val',
          'epoch', 'learning_rate'
        ],
        [
          p50, r50, ap50, ap75, map,
          epoch, args.learning_rate_fn(state.step)
        ]
      )
      logs.writer_tensorboard(writer, global_step)
      predictor.reset()
      
      ### Save weights ###
      if epoch % args.save_weights_freq == 0:
        save_weight(state)
  writer.close()
