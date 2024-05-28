#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb \
#   --total-epochs 20 --batch-size 16 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
#   --name "StARformer_3L_v0.8_golem_ai" --pred-card-idx --random-interval 1 --n-step 100 \
#   --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai" \
#   $2  # 20240516 3L step100
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb \
#   --total-epochs 20 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
#   --name "StARformer_3L_pred_cls_v0.8_golem_ai" --random-interval 1 --n-step 50 \
#   --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai" \
#   $2  # 20240516 3L pred cls step50
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb \
#   --total-epochs 20 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
#   --name "StARformer_2L_v0.8_golem_ai" --pred-card-idx --random-interval 1 --n-step 30 \
#   --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai" \
#   $2  # 20240516 2L step30
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb \
#   --total-epochs 20 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
#   --name "StARformer_no_delay_2L_v0.8_golem_ai" --pred-card-idx --random-interval 1 --n-step 50 \
#   --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai" \
#   $2  # 20240520 2L no delay step50
CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb \
  --total-epochs 20 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
  --name "StARformer_3L_v0.8_golem_ai_interval2" --pred-card-idx --random-interval 2 --n-step 50 \
  --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai" \
  $2  # 20240520 3L interval2 step50
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb \
#   --total-epochs 20 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
#   --name "DT_4L_v0.8_golem_ai" --pred-card-idx --random-interval 1 --n-step 50 \
#   --replay-dataset "/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai" \
#   $2  # 20240519 DT 4L step50
