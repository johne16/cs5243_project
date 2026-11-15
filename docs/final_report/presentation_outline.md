# Final Presentation Outline

8 to 10 minutes.

## 1. Motivation (1 min)

- Box annotation is expensive
- Detection accuracy falls when labeled data is scarce
- Question: how much does masked-image pretraining on unlabeled images help, and how does that change as labels shrink
- Hypothesis: AP drops for both pretrained and scratch as the labeled fraction shrinks, but more slowly for pretrained, so the gap widens at small fractions
- Hypothesis: both arms show the widening gap, larger for ViT because it lacks the convolutional inductive bias

## 2. Related Work (1 min)

- MAE
- ViTDet
- SparK
- Gap: no shared protocol, no sweep over labeled-set size, pretraining on outside data

## 3. Methodology (2 min)

- Two arms: ViT-Tiny with MAE and ViTDet, ResNet-18 with SparK and Faster R-CNN with FPN
- Pretraining on the same COCO images used for detection
- Pretrained run and scratch run at each labeled fraction, 10% to 100%
- Only starting weights and layer-wise learning-rate decay differ between the two runs

## 4. Experiments (1.5 min)

- Dataset: 9-class COCO 2017 subset, 18,542 training images, val2017 for evaluation
- Nested labeled fractions
- Schedules: ViT 100 epochs, ResNet-18 12 epochs
- Hardware: 2 V100 GPUs per run
- Metrics: AP, AP50, AP75, APs, APm, APl

## 5. Results (2.5 min)

- Final AP against labeled fraction, pretrained and scratch, per arm
- AP over training
- AP50, AP75, and size breakdown

## 6. Limitations (1 min)

- One seed per run
- Tiny backbones and reduced dataset
- ViT batch size reduced from 64 to 16

## 7. Future Work (0.5 min)

- Repeated seeds
- Larger backbones
- Full COCO
