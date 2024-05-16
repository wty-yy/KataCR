#!/bin/bash

python eval.py --load-epoch 4 --model-name "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240513_114808"
python eval.py --load-epoch 6 --model-name "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240513_114808"