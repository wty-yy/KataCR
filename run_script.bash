#!/bin/bash
# ./run_script.bash 3 32 0
# CUDA_VISIBLE_DEVICES=$1 python katacr/detection/train.py --train --wandb-track --batch-size $2 --load-id $3 --train-datasize $4 --model-name $5
# CUDA_VISIBLE_DEVICES=$1 python katacr/detection/train.py --train --wandb-track $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb --total-epochs 10 --batch-size 32 --nominal-batch-size 128 \
#   --name "StARformer_v0.2_golem_ai" --replay-dataset "/data/user/zhihengwu/Coding/dataset/clash-royale-replay-dataset/golem_ai" $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb --total-epochs 30 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
#   --name "StARformer_v0.4_golem_ai" $2
CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --total-epochs 30 --batch-size 32 --nominal-batch-size 128 --cnn-mode "cnn_blocks" \
  --name "StARformer_v0.4_golem_ai" $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb --total-epochs 30 --batch-size 32 --nominal-batch-size 128 --cnn-mode "csp_darknet" \
#   --name "StARformer_v0.4_golem_ai" $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb --total-epochs 100 --batch-size 16 --nominal-batch-size 128 \
#   --name "ViDformer_v0.3_golem_ai" $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb --total-epochs 10 --batch-size 32 --nominal-batch-size 128 --name "StARformer_same_action_shuffle__random_interval" $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --wandb --learning-rate 0.0006 --total-epochs 30 --batch-size 32 --nominal-batch-size 128 --replay-dataset /data/user/zhihengwu/Coding/dataset/clash-royale-replay-dataset --name "StARformer_0.1" $2
# CUDA_VISIBLE_DEVICES=$1 python katacr/policy/offline/train.py --learning-rate 0.0006 --total-epochs 30 --batch-size 32 --nominal-batch-size 128 --replay-dataset /data/user/zhihengwu/Coding/dataset/clash-royale-replay-dataset --name "StARformer_0.1" $2
