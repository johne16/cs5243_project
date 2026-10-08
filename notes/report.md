# Report notes

## Training length

| Arm | Epochs | Upstream source |
|---|---|---|
| ViT | 100 | ViTDet `mask_rcnn_vitdet_b_100ep.py`: 184,375 iterations at batch 64 on 118k images |
| ResNet-18 | 12 | SparK `coco_R_50_FPN_CONV_1x_moco_adam.yaml`: 90,000 iterations at batch 16 on 118,287 images, which is 12.2 epochs, the detectron2 1x schedule |

Each arm keeps the epoch count of the upstream recipe its detector config was built from. The ViTDet recipe pairs its 100 epochs with large scale jitter (scale 0.1 to 2.0 at 1024 pixels). The SparK recipe uses the standard 1x fine-tuning schedule for ResNet with FPN.

Iterations are computed per run as epochs * training images / batch size, so every labeled fraction trains for the same number of epochs. The learning-rate drop points keep their upstream positions: 8/9 and 26/27 of training for ViT, 2/3 and 8/9 for ResNet-18.

The comparison in this study is pretrained against scratch within each arm, and both runs of a pair use the same schedule.

## Warmup

Both arms use 500 warmup iterations at every labeled fraction. The value is an arbitrary choice.

| Arm | Upstream warmup | Source |
|---|---|---|
| ViT | 250 iterations | ViTDet `mask_rcnn_vitdet_b_100ep.py` |
| ResNet-18 | 1000 iterations | detectron2 default, restated in SparK `coco_R_50_FPN_CONV_1x_moco_adam.yaml` |

The upstream ResNet-18 value could not be kept. At the 10% fraction the run is about 1,390 iterations and the first learning-rate drop is at iteration 927, before a 1000-iteration warmup ends.

## Class imbalance

The training pool is 3000 randomly drawn train2017 images for each of the 7 classes other than person and car, combined into 18,542 images. Person and car are not sampled directly and enter through the images drawn for the other classes.

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
