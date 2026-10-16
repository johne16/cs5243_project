#!/bin/bash
#SBATCH --job-name=detect-resnet18
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=28:00:00
#SBATCH --output=detect-resnet18-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate spark_detectron2

cd SparK/downstream_d2
# $1 is the run name, such as pretrained_10
RUN=$1 python train_net.py --config-file configs/coco_R_18_FPN_CONV_1x_moco_adam.yaml
