#!/bin/bash
#SBATCH --job-name=pretrain-spark
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=pretrain-spark-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate spark_detectron2

cd SparK/pretrain
torchrun --nproc_per_node=2 main.py --model=resnet18 --data_path=../../datasets/coco --exp_name=resnet18 --exp_dir=output
