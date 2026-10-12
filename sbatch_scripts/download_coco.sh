#!/bin/bash
#SBATCH --job-name=download-coco
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=download-coco-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate download

python scripts/download_coco.py
