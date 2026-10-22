#!/bin/bash
#SBATCH --job-name=detect-resnet18
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=28:00:00
#SBATCH --output=detect-resnet18-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate spark_detectron2

cd SparK/downstream_d2
# $1 is the run name, such as pretrained_10
RUN=$1 python train_net.py --num-gpus 2 --config-file configs/coco_R_18_FPN_CONV_1x_moco_adam.yaml
