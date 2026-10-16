#!/bin/bash
#SBATCH --job-name=detect-vit
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=28:00:00
#SBATCH --output=detect-vit-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate spark_detectron2

cd detectron2
# $1 is the run name, such as pretrained_10
RUN=$1 python tools/lazyconfig_train_net.py --num-gpus 2 --config-file projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny.py
