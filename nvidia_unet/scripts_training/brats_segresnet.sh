#!/bin/bash
task=11
task_folder='11_3d'
name='BraTS2021_train'
dataset='brats'

mkdir /results/${dataset}_results
mkdir /results/${dataset}_results/${name}_segresnet
mkdir /results/${dataset}_infer/${name}_segresnet
mkdir /results/${dataset}_results/${name}_segresnet/fold-0
echo Training $name fold-0!


export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --brats  --aug --data /data_anvar/public_datasets/${task}_3d --results /results/${dataset}_results/${name}_segresnet/fold-0 --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 5 --fold 0 --amp --gpus 1 --task $task --save_ckpt

# --resume_training --ckpt_path /results//fold-0/checkpoints/last.ckpt   

echo Save predicts $name fold-0!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --brats --exec_mode predict --task ${task}_3d --data /data_anvar/public_datasets/${task}_3d --dim 3 --fold 0 --nfolds 5 --ckpt_path /results/${dataset}_results/${name}_segresnet/fold-0/checkpoints/best*.ckpt --results /results/${dataset}_infer/${name}_segresnet --amp --tta --save_preds --train_inf


python metrics.py --path_to_pred /results/${dataset}_infer/${name}_segresnet/*fold=0_tta --path_to_target /data_anvar/public_datasets/$name/labels --out /results/${dataset}_infer/${name}_segresnet/metrics_valBraTS2021_segresnet_fold-0.csv
