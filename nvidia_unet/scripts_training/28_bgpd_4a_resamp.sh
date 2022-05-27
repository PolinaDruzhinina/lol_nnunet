#!/bin/bash
task=40
task_folder='40_3d'
name='bgpd_4a_resamp'
dataset='bgpd'

python3 ../preprocess.py --data /data/private_data/$dataset --task $task --ohe --exec_mode training --results /data/private_data/$dataset/

# mkdir /results/schw_results
mkdir /results/${dataset}_results/$name
mkdir /results/${dataset}_results/$name/fold-0
echo Training $name fold-0!
# --resume_training --ckpt_path /results/schw_results/schw_1_reg/fold-0/checkpoints/last.ckpt 

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/${dataset}/${task}_3d --results /results/${dataset}_results/$name/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 0 --amp --gpus 1 --task $task --save_ckpt

echo Save predicts $name fold-0!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task ${task}_3d --data /data/private_data/${dataset}/${task}_3d --no_back_in_output --dim 3 --fold 0 --nfolds 3 --ckpt_path /results/${dataset}_results/$name/fold-0/checkpoints/best*.ckpt --results /results/${dataset}_infer/$name --amp --tta --save_preds

# python metrics_1class.py --path_to_pred /results/${dataset}_infer/$name/*fold=0_tta --path_to_target  /data/private_data/${dataset}/$name/labels --out /results/${dataset}_infer/$name/metrics_${name}_fold-0.csv

mkdir /results/${dataset}_results/$name/fold-1
echo Training $name fold-1!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/${dataset}/${task}_3d --results /results/${dataset}_results/$name/fold-1 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 1 --amp --gpus 1 --task $task --save_ckpt

echo Save predicts $name fold-1!
mkdir /results/${dataset}_infer/$name

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task ${task}_3d --data /data/private_data/${dataset}/${task}_3d --no_back_in_output --dim 3 --fold 1 --nfolds 3 --ckpt_path /results/${dataset}_results/$name/fold-1/checkpoints/best*.ckpt --results /results/${dataset}_infer/$name --amp --tta --save_preds

# python metrics_1class.py --path_to_pred /results/${dataset}_infer/$name/*fold=1_tta --path_to_target  /data/private_data/${dataset}/$name/labels --out /results/${dataset}_infer/$name/metrics_${name}_fold-1.csv

mkdir /results/${dataset}_results/$name/fold-2
echo Training $name fold-2!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/${dataset}/${task}_3d --results /results/${dataset}_results/$name/fold-2 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 2 --amp --gpus 1 --task $task --save_ckpt

echo End Training!

echo Save predicts $name fold-2!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task ${task}_3d --data /data/private_data/${dataset}/${task}_3d --no_back_in_output --dim 3 --fold 2 --nfolds 3 --ckpt_path /results/${dataset}_results/$name/fold-2/checkpoints/best*.ckpt --results /results/${dataset}_infer/$name --amp --tta --save_preds

# python metrics_1class.py --path_to_pred /results/${dataset}_infer/$name/*fold=2_tta --path_to_target  /data/private_data/${dataset}/$name/labels --out /results/${dataset}_infer/$name/metrics_${name}_fold-2.csv
