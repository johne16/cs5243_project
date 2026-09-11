# Repo Reading Roadmap

Read each repo along its execution path: entry script, then config, then model, then the training step. Ignore everything else.

## MAE (facebookresearch/mae)

Smallest, start here.

1. `PRETRAIN.md` - the command line, which tells you which args matter.
2. `main_pretrain.py` - argument parsing, dataset, model construction, epoch loop.
3. `models_mae.py` - the whole method. `random_masking`, `forward_encoder`, `forward_decoder`, `forward_loss`. This one file is the paper.
4. `engine_pretrain.py` - one training epoch, about 50 lines.
5. `models_vit.py` and `main_finetune.py` - how the pretrained encoder is reused without the decoder.

## SparK (keyu-tian/SparK)

Same shape.

1. `pretrain/README.md`, then `pretrain/main.py`.
2. `pretrain/spark.py` - the masking and the reconstruction loss.
3. `pretrain/encoder.py` - where dense convolutions become sparse so masked regions stay empty. This is the part that differs from MAE.
4. `pretrain/models/resnet.py` - the ResNet modified to work under masking.
5. `downstream_d2/` - `convert-timm-to-d2.py` and `train_net.py`. This is the bridge to Detectron2, and the reason a ResNet backbone is easier than ConvNeXt here.

## Detectron2 / ViTDet (facebookresearch/detectron2)

Biggest, read last and least.

1. `projects/ViTDet/README.md`, then one config: `projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py`, which layers onto `configs/common/models/mask_rcnn_vitdet.py`. Detectron2 configs are Python that builds objects, so the config is a readable model definition.
2. `tools/lazyconfig_train_net.py` - the entry point those configs run under.
3. `detectron2/modeling/backbone/vit.py` - the ViTDet backbone, including how a plain Vision Transformer produces multi-scale features.
4. `detectron2/modeling/meta_arch/rcnn.py` - the detector skeleton, shared by ViTDet and Faster R-CNN.
5. `detectron2/modeling/backbone/resnet.py` - only when wiring in the SparK weights.

## Order

MAE fully, SparK fully, then Detectron2 only as far as the two backbone files. The first two repos teach the method; the third is plumbing you configure rather than read.
