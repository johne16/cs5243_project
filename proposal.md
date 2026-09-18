# Masked Pretraining vs Random Initialization for Object Detection

## Problem Statement

Object detectors are normally trained the classical supervised way: every training image carries human-drawn boxes, and the whole network learns from that signal alone. Box annotation is the expensive part of that pipeline, and accuracy falls off sharply when the labeled set is small. Masked-image pretraining offers an alternative source of signal. The backbone first learns from unlabeled images by hiding most of each image and reconstructing what was hidden, and only then is the detector fine-tuned on labeled boxes. What is not clear from a practitioner's standpoint is how much detection accuracy that unlabeled stage actually buys, how that benefit scales as labels become scarce, and whether it behaves the same way for a transformer backbone and a convolutional one.

## Background and Related Work

MAE (He et al., 2022) hides 75% of the image patches, drops them entirely, encodes only the visible ones, and reconstructs the missing pixels. ViTDet (Li et al., 2022) showed that a plain, non-hierarchical Vision Transformer pretrained this way is a competitive detection backbone without any pyramid design. SparK (Tian et al., 2023) is the convolutional counterpart, using sparse convolutions and a hierarchical decoder so a plain convolutional network can be pretrained under the same masked objective. Each of these papers reports its own gains against its own baselines, under different schedules, augmentation, and detection heads, and both pretrain the backbone on ImageNet-1K images before fine-tuning the detector on COCO, so the two stages see two different image sets. This project places both in one shared protocol, pretrains on the same COCO images that the detector is later trained on so that no outside data enters the comparison, and adds an explicit sweep over labeled-set size, which neither paper reports.

## Research Question / Hypothesis

How much does masked-image pretraining on unlabeled COCO images improve object-detection accuracy over training from random initialization, and how does that gain change as the number of labeled training images shrinks, for both a transformer (ViTDet + MAE) and a CNN (Faster R-CNN + SparK)?

Hypothesis: Average Precision (AP) drops for both the pretrained and the randomly initialized detector as the labeled fraction shrinks, but it drops more slowly for the pretrained one, so the AP gap between them widens at the smaller fractions. Both backbone families are expected to show that widening gap, with the transformer showing the larger one because it lacks the convolutional inductive bias and so depends more on what pretraining supplies.

## Proposed Approach

Two detector and pretraining pairs are studied:

| Detector | Backbone | Pretraining method for that backbone |
|---|---|---|
| ViTDet (Detectron2) | ViT-Tiny | MAE |
| Faster R-CNN (Detectron2) | ResNet-18 | SparK |

Two A100 GPUs are available, so the backbones are the tiny variants of each family rather than the sizes the original papers used. Neither variant ships as a released config: the MAE and ViTDet repositories provide ViT-B, ViT-L, and ViT-H, and SparK provides ResNet-50 and larger, so the tiny configurations are written by hand by setting embedding dimension, depth, and head count for the transformer and selecting the shallower ResNet for the convolutional side.

Pretraining produces backbone weights only; the detection heads are always randomly initialized. The second stage, detection training, is run twice at each labeled fraction from 10% to 100% in steps of 10%: once starting from the pretrained backbone, which is fine-tuning, and once starting from a randomly initialized backbone, which is training from scratch. Nothing else changes between the two runs, so the difference in final accuracy is what the pretraining bought. Both detectors run in Detectron2, so the detection head, augmentation, schedule, and evaluation code are shared across every run.

## Datasets

COCO 2017: 118k training images, 5k validation, 80 classes, standard 2D box annotations, open download with no approval process (https://cocodataset.org/). Compute limits make the full dataset impractical, so the study is run on a reduced version of it. The class list is narrowed to a fixed subset of the 80 categories, and a fixed random subset of the train2017 images containing those categories is drawn and used for the entire study. The same class list is applied to the validation annotations, so no category is scored that was never trained. The same subset serves both stages: pretraining uses the images with annotations withheld, detection training uses the annotations. The labeled fractions are drawn as nested subsets of that pool, so the 10% split is contained in the 20% split and so on, and 100% means the full subset. Evaluation is on val2017 in every case, restricted to the same class list used for training.

## Evaluation Metrics

COCO AP, averaged over the ten IoU thresholds from 0.50 to 0.95 and over the retained classes, is the primary metric.

Four secondary numbers are reported alongside it. AP50 and AP75 hold the IoU threshold fixed at 0.50 and 0.75; the first is a loose criterion that mostly measures whether the object was found, the second is strict and measures how tightly the box is drawn, so the spread between them separates recognition failures from localization failures. AP_S, AP_M, and AP_L restrict the same score to ground-truth objects in three size bins, split on the `area` field that every COCO instance annotation carries, which is the pixel count inside the object's outline rather than the area of its box: small is under 1024 square pixels, medium is 1024 to 9216, large is above 9216. This breakdown shows whether any gain from pretraining is spread evenly across object scales or concentrated in one size range.

The headline result is the AP difference between the pretrained and randomly initialized run at each labeled fraction, plotted as a curve of that difference against labeled-set size for each backbone family.

## Expected Challenges

Compute is the binding constraint. Masked pretraining and detection training are both long jobs, and the design calls for two pretraining runs plus forty detection runs, which is what forces the tiny backbones, the short schedules, and the reduced image pool, and puts the absolute AP numbers well below published values. The two arms are not the same kind of job: the pretrained arm is fine-tuning an already useful backbone, while the random arm is training one from scratch, and training from scratch is known to need a far longer schedule to converge. A schedule cut short therefore overstates the benefit of pretraining, so the budget has to be generous enough that the from-scratch arm is not simply undertrained, and the AP curve over training is reported so that undertraining is visible rather than assumed. At the 10% fraction, run-to-run variance from the particular images sampled can rival the effect being measured, so each small-fraction setting needs repeated runs with different seeds. SparK and MAE also come from separate codebases with their own conventions for normalization, learning-rate scaling, and layer-wise decay, and porting both into Detectron2 without silently changing one of them is a real integration risk.
