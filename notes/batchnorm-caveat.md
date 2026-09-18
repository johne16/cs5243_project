# BatchNorm configuration for the ResNet arm

Detectron2 defaults `MODEL.RESNETS.NORM` to `FrozenBN`. Frozen batch norm keeps the statistics and the affine parameters fixed at whatever the loaded checkpoint carried. That is only valid when the backbone comes from a large pretrained checkpoint.

Both ResNet arms in this project break that assumption:

- The random-init arm has no statistics to freeze.
- The SparK-pretrained arm has statistics from the masked pretraining run, not from ImageNet supervised training.

Set `MODEL.RESNETS.NORM` to `SyncBN` (two GPUs) or `GN` for both arms, and use the same setting in both so the comparison stays matched.

Separate issue on the pretraining side: SparK masks out most of the input, so batch norm statistics have to be computed over the unmasked positions only. Check how the SparK repo handles this before porting the backbone into Detectron2.

Pixel normalization (`MODEL.PIXEL_MEAN`, `MODEL.PIXEL_STD`, `INPUT.FORMAT`) is a separate matter and just needs the MAE, SparK, and Detectron2 values set consistently.
