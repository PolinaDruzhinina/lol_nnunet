#!/bin/bash
task=11_f
task_folder='11_f_3d'
name='BraTS2021_flair'
dataset='brats'
data_path='/home/jovyan/datasets/brats_preproc'
python3 ../preprocess.py --data $data_path --task $task --ohe --exec_mode training --results $data_path

# mkdir /results/schw_results
mkdir /home/jovyan/polina/experiments/results/${dataset}
mkdir /home/jovyan/polina/experiments/results/${dataset}/${name}
mkdir /home/jovyan/polina/experiments/results/${dataset}/${name}/fold-0
echo Training $name fold-0!


export CUDA_VISIBLE_DEVICES=0 && python ../main.py --data $data_path/${task}_3d --results /home/jovyan/polina/experiments/results/${dataset}/${name}/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 5 --fold 0 --brats --aug --amp --gpus 1 --task $task --save_ckpt


export CUDA_VISIBLE_DEVICES=1 && python ../main.py --brats --exec_mode predict --task ${task}_3d --data $data_path/${task}_3d --dim 3 --fold 0 --nfolds 5 --ckpt_path /home/jovyan/polina/experiments/results/${dataset}/${name}/fold-0/checkpoints/best*.ckpt --results /home/jovyan/polina/experiments/${dataset}_infer/$name --amp --tta --save_preds --train_inf

python metrics.py --path_to_pred /home/jovyan/polina/experiments/${dataset}_infer/$name/*fold=0_tta --path_to_target  /$data_path/$name/labels --out /home/jovyan/polina/experiments/${dataset}_infer/$name/metrics_valBraTS2021_f_fold-0.csv

