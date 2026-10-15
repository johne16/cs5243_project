# Masked Pretraining vs Random Initialization for Object Detection

## Progress Completed

### Dataset

A 9-class subset of COCO 2017 is downloaded: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat.

- Training pool: 3000 randomly drawn train2017 images for each of the 7 classes other than person and car, combined into 18,542 images.
- Validation: every val2017 image containing at least one of the 9 classes.
- Annotation files keep only the 9 classes.

| Class | Images in pool |
|---|---|
| person | 11,567 |
| bicycle | 3,073 |
| car | 5,378 |
| motorcycle | 3,124 |
| airplane | 2,986 |
| bus | 3,280 |
| train | 3,046 |
| truck | 4,151 |
| boat | 3,002 |

### Code

| Component | Status |
|---|---|
| ViT-Tiny MAE model (192 dim, 12 layers, 3 heads, patch 16) | Added to the MAE fork |
| ResNet-18 SparK model | Registered in the SparK fork |
| ViTDet Faster R-CNN config for ViT-Tiny, 9 classes, no mask branch | Written |
| Faster R-CNN with FPN config for ResNet-18, 9 classes, no mask branch | Written |
| Labeled-fraction dataset registration | Written, shared by both arms |
| Per-run settings files, 20 runs per arm | Written |
| Schedule computed per run from epochs, training images, and batch size | Written for both arms |
| Conda environments and Slurm job scripts | Written |

The labeled fractions are drawn from one fixed shuffle of the training pool, so every smaller fraction is a subset of every larger one, and both arms train on the same images at each fraction.

| Arm | Epochs | Batch size | Warmup | LR drop points |
|---|---|---|---|---|
| ViT | 100 | 64 | 500 iterations | 8/9 and 26/27 of training |
| ResNet-18 | 12 | 16 | 500 iterations | 2/3 and 8/9 of training |

### Pretraining

| Arm | Status |
|---|---|
| MAE, ViT-Tiny | Finished |
| SparK, ResNet-18 | Running |

## Preliminary Results

MAE pretraining of ViT-Tiny on the 18,542-image training pool ran 400 epochs on 1 V100 GPU in 5.9 hours, with batch size 64, mask ratio 0.75, and 40 warmup epochs.

| Epoch | Reconstruction loss |
|---|---|
| 0 | 1.686 |
| 10 | 0.543 |
| 50 | 0.433 |
| 100 | 0.409 |
| 200 | 0.391 |
| 300 | 0.380 |
| 399 | 0.374 |

No detection runs have finished yet.

## Challenges Encountered

- MAE requires `timm==0.3.2` and SparK requires `timm==0.5.4`, so the two pretraining codebases run in separate conda environments.
- Building the detectron2 fork on the cluster requires loading gcc 9.3.0 and the CUDA 11.3 toolkit.
- Neither codebase ships a configuration for the tiny backbones. The ViT-Tiny MAE model, the ResNet-18 SparK registration, and both detection configs were written by hand.
- The built-in COCO dataset registrations in detectron2 carry 80-class metadata that conflicts with the 9-class annotation files, so both arms register their own training and validation datasets.

## Remaining Work

- Finish SparK pretraining of ResNet-18.
- Run the 40 detection runs: pretrained and from scratch, at labeled fractions 10% to 100%, for both arms.
- Plot the AP difference between pretrained and from-scratch runs against labeled fraction for each arm.
- Write the final report.
