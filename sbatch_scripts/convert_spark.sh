#!/bin/bash
#SBATCH --job-name=convert-spark
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=convert-spark-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate spark_detectron2

cd SparK/downstream_d2
python convert-timm-to-d2.py ../pretrain/output/resnet18_1kpretrained_timm_style.pth ../pretrain/output/resnet18_1kpretrained_d2_style.pkl
