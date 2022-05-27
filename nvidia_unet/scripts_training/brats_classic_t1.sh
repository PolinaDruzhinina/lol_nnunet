#!/bin/bash
task=11_t1
task_folder='11_t1_3d'
name='BraTS2021_t1'
dataset='brats'
python3 ../preprocess.py --data /data/public_data --task $task --ohe --exec_mode training --results /data/public_data

mkdir /results/${dataset}_results
mkdir /results/${dataset}_results/${name}
mkdir /results/${dataset}_results/${name}/fold-0
echo Training $name fold-0!


export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --data /data/public_data/${task}_3d --results /results/${dataset}_results/${name}/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 5 --fold 0 --aug --brats --amp --gpus 1 --task $task --resume_training --ckpt_path /results/brats_results/BraTS2021_t1/fold-0/checkpoints/last.ckpt  

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --brats --exec_mode predict --task ${task}_3d --data /data/public_data/${task}_3d --dim 3 --fold 0 --nfolds 5 --ckpt_path /results/${dataset}_results/${name}/fold-0/checkpoints/best*.ckpt --results /results/${dataset}_infer/$name --amp --tta --save_preds --train_inf

python metrics.py --path_to_pred /results/${dataset}_infer/$name/*fold=0_tta --path_to_target  /data/public_data/$name/labels --out /results/${dataset}_infer/$name/metrics_valBraTS2021_t1_fold-0.csv

