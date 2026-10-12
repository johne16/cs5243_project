# Pipeline review

## Depends on the server environments

- [ ] 4. NumPy 1.24 or newer breaks two places.
  - `SparK/downstream_d2/train_net.py:198-201` uses `np.int` and `np.bool` on every training start.
  - `mae/util/pos_embed.py:56` uses `np.float` when the MAE model is built.

- [ ] 5. MAE needs old torch and timm.
  `mae/util/misc.py:21` imports `torch._six`, which was removed in torch 2.0. `mae/main_pretrain.py:27` asserts timm is exactly 0.3.2.

- [ ] 6. The MAE checkpoint may not load in detectron2 on torch 2.6 or newer.
  The checkpoint stores a pickled `args` object, and `torch.load` there defaults to weights-only loading, which rejects it. The load call is in fvcore, which is not in the repo. Not confirmed.

## No bugs found in

- `scripts/download_coco.py`, checked against the installed fiftyone 1.22.1 source
- `detectron2/detectron2/data/datasets/labeled_fraction.py`
- `faster_rcnn_vitdet_tiny.py` and its runs yaml
- the ResNet-18 yaml and its runs yaml
- `setup()` in `SparK/downstream_d2/train_net.py`
- `SparK/downstream_d2/lr_decay.py`
- `mae_vit_tiny_patch16`
- `SparK/downstream_d2/convert-timm-to-d2.py` key mapping for ResNet-18
