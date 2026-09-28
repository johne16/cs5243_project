# Masked Pretraining vs Random Initialization for Object Detection

## Layout

| Folder | Contents |
|---|---|
| `detectron2/` | Fork of facebookresearch/detectron2. Faster R-CNN detection training for the ViT-Tiny and ResNet-18 backbones. |
| `mae/` | Fork of facebookresearch/mae. MAE pretraining for the ViT-Tiny backbone. |
| `SparK/` | Fork of keyu-tian/SparK. SparK pretraining for the ResNet-18 backbone. |
| `notes/` | Reading notes. |
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

## MAE pretraining

From `mae/`:

```
python main_pretrain.py --model mae_vit_tiny_patch16 --data_path <data_path> --output_dir <output_dir> --log_dir <output_dir>
```

## SparK pretraining

From `SparK/pretrain/`:

```
python main.py --model=resnet18 --data_path=<data_path> --exp_name=<exp_name> --exp_dir=<exp_dir>
```

## ViTDet training runs

Run settings are in `detectron2/projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny_runs.yaml`, one entry per run. `RUN` selects the entry.

From `detectron2/`:

```
RUN=pretrained_10 python tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/COCO/faster_rcnn_vitdet_tiny.py
```

## ResNet-18 training runs

Run settings are in `SparK/downstream_d2/configs/coco_R_18_FPN_CONV_1x_moco_adam_runs.yaml`, one entry per run. `RUN` selects the entry.

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
