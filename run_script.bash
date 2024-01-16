#!/bin/bash
# ./run_script.bash 3 32 0
CUDA_VISIBLE_DEVICES=$1 python katacr/detection/train.py --train --wandb-track --batch-size $2 --load-id $3 --train-datasize $4 --model-name $5
