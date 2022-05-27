#!/bin/bash
task=11
task_folder='11_3d'
name='BraTS2021_train'
dataset='brats'

mkdir /results/${dataset}_results
mkdir /results/${dataset}_results/${name}_no_aug
mkdir /results/${dataset}_results/${name}_no_aug/fold-0
echo Training $name fold-0!


export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --data /data_anvar/public_datasets/${task}_3d --results /results/${dataset}_results/${name}_no_aug/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 5 --fold 0 --brats --amp --gpus 1 --task $task --save_ckpt

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --brats --exec_mode predict --task ${task}_3d --data /data/private_data/${dataset}/${task}_3d --dim 3 --fold 0 --nfolds 5 --ckpt_path /results/${dataset}_results/${name}_unetr/fold-0/checkpoints/best*.ckpt --results /results/${dataset}_infer/$name --amp --tta --save_preds

python metrics.py --path_to_pred /results/brats_infer/BraTS2021_train_no_aug/*fold=0_tta --path_to_target  /data_anvar/public_datasets/BraTS2021_train/labels --out /results/brats_infer/BraTS2021_train_no_aug/metrics_valBraTS2021_train_no_aug_fold-0.csv

# --resume_training --ckpt_path /results/fold-0/checkpoints/last.ckpt   
