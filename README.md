# Masked Pretraining vs Random Initialization for Object Detection

## Layout

| Folder | Contents |
|---|---|
| `detectron2/` | Fork of facebookresearch/detectron2. Faster R-CNN detection training for the ViT-Tiny and ResNet-18 backbones. |
| `mae/` | Fork of facebookresearch/mae. MAE pretraining for the ViT-Tiny backbone. |
| `SparK/` | Fork of keyu-tian/SparK. SparK pretraining for the ResNet-18 backbone. |
| `scripts/` | Dataset download script and its requirements. |
| `datasets/` | Downloaded dataset. Not tracked. |
| `docs/` | Project web page. |
| `notes/` | Reading notes and report notes. |
| `proposal.md` | Project proposal. |

The three forks are git submodules. The root repo records which commit of each fork to use.

## Clone

```
git clone --recurse-submodules <root-repo-url>
```

If the root repo is already cloned without the submodules:

```
git submodule update --init --recursive
```

## Pull latest

```
git pull
git submodule update --init --recursive
```

## Change code in a fork

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

## Dataset

The study uses a subset of COCO 2017 with 9 classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat.

- Training pool: 3000 randomly drawn train2017 images for each of the 7 classes other than person and car, combined.
- Validation: every val2017 image that contains at least one of the 9 classes.
- Annotations are kept for the 9 classes only.

The download script has its own environment, separate from the training environments. It needs Python 3.10 or newer. From `scripts/`, create and activate a venv, then:

```
pip install -r requirements.txt
```

Download from the repo root:

```
python scripts/download_coco.py
```

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

## MAE pretraining

From `mae/`:

```
python main_pretrain.py --model mae_vit_tiny_patch16 --data_path ../datasets/coco --output_dir <output_dir> --log_dir <output_dir>
```

## SparK pretraining

From `SparK/pretrain/`:

```
python main.py --model=resnet18 --data_path=../../datasets/coco --exp_name=<exp_name> --exp_dir=<exp_dir>
```

## Detection training

Both detection arms import from the `detectron2/` fork, so the fork is the detectron2 that must be installed.

Each run trains on the first `labeled_fraction` of a fixed shuffle of the training pool, so a smaller fraction is a subset of a larger one. The number of iterations is computed from the number of training images in the run.

| Arm | Epochs | Batch size | Warmup | Evaluation and checkpoint period |
|---|---|---|---|---|
| ViT | 100 | 64 | 500 iterations | 500 iterations |
| ResNet-18 | 12 | 16 | 500 iterations | 500 iterations |

## ViTDet training runs

Run settings are in `detectron2/projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny_runs.yaml`, one entry per run. `RUN` selects the entry. The dataset folder is `_coco_dir` in `faster_rcnn_vitdet_tiny.py`, relative to `detectron2/`.

From `detectron2/`:

```
RUN=pretrained_10 python tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny.py
```

## ResNet-18 training runs

Run settings are in `SparK/downstream_d2/configs/coco_R_18_FPN_CONV_1x_moco_adam_runs.yaml`, one entry per run. `RUN` selects the entry. The dataset folder is `DATASETS.COCO_DIR` in `coco_R_18_FPN_CONV_1x_moco_adam.yaml`, relative to `SparK/downstream_d2/`.

From `SparK/downstream_d2/`:

```
RUN=pretrained_10 python train_net.py --config-file configs/coco_R_18_FPN_CONV_1x_moco_adam.yaml
```

## Pull upstream changes into a fork

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
