#!/bin/bash

python eval.py --load-epoch 11 --eval-num 4 --model-name "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step30__0__20240516_125734"
python eval.py --load-epoch 1 --model-name "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step30__0__20240516_125734"
python eval.py --load-epoch 2 --model-name "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step30__0__20240516_125734"
python eval.py --load-epoch 4 --model-name "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step30__0__20240516_125734"