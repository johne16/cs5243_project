#!/bin/bash
#SBATCH --job-name=pretrain-mae
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=pretrain-mae-%j.log

source /apps/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate mae

# main_pretrain.py switches to distributed mode when SLURM_PROCID is set
unset SLURM_PROCID

cd mae
python main_pretrain.py --model mae_vit_tiny_patch16 --data_path ../datasets/coco --output_dir output --log_dir output
