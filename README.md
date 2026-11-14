# Masked Pretraining vs Random Initialization for Object Detection

Measures how much masked-image pretraining improves COCO object detection over random initialization, at labeled fractions from 10% to 100%.

| Arm | Backbone | Pretraining | Detector |
|---|---|---|---|
| ViT | ViT-Tiny | MAE | ViTDet with Faster R-CNN heads |
| ResNet-18 | ResNet-18 | SparK | Faster R-CNN with FPN |

## Layout

| Folder | Contents |
|---|---|
| `detectron2/` | Fork of facebookresearch/detectron2. Detection training for both arms. |
| `mae/` | Fork of facebookresearch/mae. MAE pretraining for ViT-Tiny. |
| `SparK/` | Fork of keyu-tian/SparK. SparK pretraining for ResNet-18 and the ResNet-18 detection script. |
| `scripts/` | Dataset download script and results compile script. |
| `sbatch_scripts/` | Slurm job scripts for every step. |
| `logs/` | Detection training logs, in `pretrained/` and `scratch/`. |
| `results/` | Compiled results. |
| `datasets/` | Downloaded dataset. Not tracked. |
| `docs/` | Project web page, proposal, midterm presentation, and final presentation outline. |
| `notes/` | Reading notes and report notes. |

The three forks are git submodules. The root repo records which commit of each fork to use.

## Setup

### Hardware

All training ran as Slurm jobs. MAE pretraining used 1 V100 GPU. SparK pretraining and every detection run used 2 V100 GPUs with 32 GB each.

### Clone

```
git clone --recurse-submodules <root-repo-url>
```

If the root repo is already cloned without the submodules:

```
git submodule update --init --recursive
```

### Environments

| Environment | File | Used for |
|---|---|---|
| `download` | `environment_download.yml` | Dataset download |
| `mae` | `environment_mae.yml` | MAE pretraining |
| `spark_detectron2` | `environment_spark_detectron2.yml` | SparK pretraining, both detection arms |

MAE needs `timm==0.3.2` and SparK needs `timm==0.5.4`, so they use separate environments. Create all three from the repo root:

```
conda env create -f environment_download.yml
conda env create -f environment_mae.yml
conda env create -f environment_spark_detectron2.yml
```

Install the detectron2 fork into `spark_detectron2`. Both detection arms import from it. The build needs gcc and a CUDA 11.3 toolkit.

```
module load gcc/9.3.0 cudatoolkit/11.3.1_465.19.01
conda activate spark_detectron2
pip install -e detectron2
```

## Reproducing the experiments

Run the steps in order. Each step lists its Slurm job and the command the job runs.

> [!IMPORTANT]
> Submit every job from the repo root. The scripts use paths relative to it and fail from any other folder.

### 1. Download the dataset

```
sbatch sbatch_scripts/download_coco.sh
```

Runs `python scripts/download_coco.py` from the repo root in the `download` environment.

The study uses a subset of COCO 2017 with 9 classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat.

- Training pool: 3000 randomly drawn train2017 images for each of the 7 classes other than person and car, combined.
- Validation: every val2017 image that contains at least one of the 9 classes.
- Annotations are kept for the 9 classes only.

The script writes:

```
datasets/coco/
  train/data/
  validation/data/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  raw/
  scratch/
```

`raw/` holds the full original annotation files and `scratch/` holds the downloaded annotation zip.

### 2. MAE pretraining

```
sbatch sbatch_scripts/pretrain_mae.sh
```

Runs, from `mae/`:

```
python main_pretrain.py --model mae_vit_tiny_patch16 --data_path ../datasets/coco --output_dir output --log_dir output
```

Trains for the default 400 epochs. The ViT pretrained runs load the last checkpoint, `mae/output/checkpoint-399.pth`.

### 3. SparK pretraining

```
sbatch sbatch_scripts/pretrain_spark.sh
```

Runs, from `SparK/pretrain/`:

```
torchrun --nproc_per_node=2 main.py --model=resnet18 --data_path=../../datasets/coco --exp_name=resnet18 --exp_dir=output
```

Trains for the default 1600 epochs. If the job stops at its time limit, continue it with:

```
sbatch sbatch_scripts/pretrain_spark_resume.sh
```

Then convert the backbone to the detectron2 format:

```
sbatch sbatch_scripts/convert_spark.sh
```

Runs, from `SparK/downstream_d2/`:

```
python convert-timm-to-d2.py ../pretrain/output/resnet18_1kpretrained_timm_style.pth ../pretrain/output/resnet18_1kpretrained_d2_style.pkl
```

The ResNet-18 pretrained runs load `SparK/pretrain/output/resnet18_1kpretrained_d2_style.pkl`.

### 4. Detection training

Each arm has 20 runs: `pretrained_10` to `pretrained_100` and `scratch_10` to `scratch_100`, in steps of 10. Each job trains one run, given by name after the script:

```
sbatch sbatch_scripts/detect_vit.sh pretrained_10
sbatch sbatch_scripts/detect_resnet18.sh pretrained_10
```

Each run trains on the first `labeled_fraction` of a fixed shuffle of the training pool, so a smaller fraction is a subset of a larger one. The number of iterations is computed from the number of training images in the run.

| Arm | Epochs | Batch size | Warmup | Evaluation and checkpoint period |
|---|---|---|---|---|
| ViT | 100 | 16 | 500 iterations | 500 iterations |
| ResNet-18 | 12 | 16 | 500 iterations | 500 iterations |

| Arm | Command, run from | Runs file | Output |
|---|---|---|---|
| ViT | `detectron2/` | `detectron2/projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny_runs.yaml` | `detectron2/output/<RUN>/` |
| ResNet-18 | `SparK/downstream_d2/` | `SparK/downstream_d2/configs/coco_R_18_FPN_CONV_1x_moco_adam_runs.yaml` | `SparK/downstream_d2/output/<RUN>/` |

ViT:

```
RUN=pretrained_10 python tools/lazyconfig_train_net.py --num-gpus 2 --config-file projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny.py --resume
```

ResNet-18:

```
RUN=pretrained_10 python train_net.py --num-gpus 2 --config-file configs/coco_R_18_FPN_CONV_1x_moco_adam.yaml
```

`RUN` selects the entry in the runs file. The dataset folder is `_coco_dir` in `faster_rcnn_vitdet_tiny.py` for ViT and `DATASETS.COCO_DIR` in `coco_R_18_FPN_CONV_1x_moco_adam.yaml` for ResNet-18.

`--resume` loads the last checkpoint in the run's output folder. `detect_vit.sh` passes it, so resubmitting the same job continues a ViT run stopped at its time limit.

### 5. Compile results

Move the detection logs into `logs/pretrained/` and `logs/scratch/`, then from the repo root:

```
python scripts/compile_results.py
```

Writes every evaluation to `results/results.csv`.

## Git workflow

### Pull latest

```
git pull
git submodule update --init --recursive
```

### Change code in a fork

Commit and push inside the fork, then record the new fork commit in the root repo. `git submodule update` leaves each fork on a detached commit, so check out `main` before making changes.

```
cd detectron2
git checkout main
git pull
git add <files>
git commit -m "<message>"
git push
cd ..
git add detectron2
git commit -m "<message>"
git push
```

### Pull upstream changes into a fork

Add the upstream remote once per fork:

```
git -C detectron2 remote add upstream https://github.com/facebookresearch/detectron2.git
git -C mae remote add upstream https://github.com/facebookresearch/mae.git
git -C SparK remote add upstream https://github.com/keyu-tian/SparK.git
```

Then:

```
cd detectron2
git checkout main
git fetch upstream
git merge upstream/main
git push
```
