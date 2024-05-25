#!/bin/bash

python eval.py --load-epoch 3 --eval-num 100 --model-name "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240512_181646"
# python eval.py --load-epoch 4 --eval-num 20 --model-name "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step30__0__20240512_181548"

# Evaluate all models
# model_name=(
#   "DT_4L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240519_224135"
#   "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step30__0__20240516_125734"
#   "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240513_114808"
#   "StARformer_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step100__0__20240513_114949"
#   "StARformer_3L_pred_cls_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240516_125201"
#   "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step100__0__20240516_124617"
#   "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step30__0__20240512_181548"
#   "StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240512_181646"
#   "StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252"
# )
# for name in ${model_name[@]}
# do
#   for i in {1..10}
#   do
#     python eval.py --load-epoch $i --eval-num 20 --model-name $name
#   done
# done