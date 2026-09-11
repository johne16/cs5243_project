# Masked Pretraining vs Supervised Training for Object Detection

## The idea

Object detectors are normally trained the classical supervised way: every training image carries human-drawn boxes, and the whole network learns from that signal alone. Masked autoencoding is the alternative. The backbone first learns from unlabeled images by hiding most of each image and reconstructing what was hidden, and only then is the detector fine-tuned on labeled boxes.

The project trains both ways under one protocol and measures what the masked pretraining actually buys, especially when the labeled set is small.

## Data

**COCO 2017**. 118k training images, 5k validation, 80 classes, standard 2D box annotations. Open download, no approval process.
https://cocodataset.org/

The same images serve both regimes. Masked pretraining uses the images with the annotations withheld, supervised training uses the annotations.

## Detectors and pretraining methods

Every run in this project is an object detector: an image goes in, boxes and class labels come out. A detector is built from a backbone, which turns pixels into features, and a set of detection heads on top of it, which propose and classify boxes.

MAE and SparK are pretraining methods. Each one trains a backbone on unlabeled images by hiding part of the input and reconstructing it, and the only thing that survives is a set of backbone weights. The detection heads do not exist during pretraining. They start at random values, and everything trains together on labeled boxes from the first step on.

| Detector | Backbone | Pretraining method for that backbone |
|---|---|---|
| [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) (Detectron2) | plain Vision Transformer | [MAE](https://github.com/facebookresearch/mae) |
| [Faster R-CNN](https://github.com/facebookresearch/detectron2) (Detectron2) | ResNet-50 | [SparK](https://github.com/keyu-tian/SparK) |

MAE hides 75% of the image patches, drops them entirely, encodes only the visible ones, and reconstructs the missing pixels. SparK is the convolutional counterpart, using sparse convolutions so a plain convolutional network can be pretrained the same way. Pairing the two means the comparison is not confined to transformers.

Each detector is trained twice: once with its backbone starting from the pretrained weights, once with the backbone starting from random values. Nothing else changes between the two runs, so the difference in final accuracy is what the pretraining bought.

Both run in Detectron2, so the detection head, augmentation, schedule, and evaluation code are shared across every run.
