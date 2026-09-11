# Training Concepts and Shape Traces

Every trained model here is a detector: image in, boxes and class labels out. MAE and SparK are not detectors. They only produce backbone weights, which are loaded into the detector's backbone before training starts, while the pyramid and heads begin at random values.


## Path A: MAE pretraining -> ViTDet fine-tuning

### Concepts (MAE pretraining)

In order of execution.

1. **Data loading.** Take a random region covering 20% to 100% of the image, resize it to 224 x 224 with bicubic interpolation, flip left-right at random, then subtract a fixed per-channel mean and divide by a fixed per-channel standard deviation. Both repos use the constants computed from ImageNet, which are standard regardless of the dataset being trained on. Labels are never read.
2. **Patchify.** Cut the image into a grid of non-overlapping p x p patches and flatten each patch into a vector. Reversible: `unpatchify` puts pixels back.
3. **Patch embedding.** One Conv2d with kernel = stride = patch size projects each patch to the model width. Equivalent to patchify plus a linear layer.
4. **Fixed 2D sin-cos position embedding.** Not learned. Stored as a `requires_grad=False` parameter and filled once at init. A separate learned CLS token is prepended.
5. **Random masking by rank.** Draw uniform noise per patch, `argsort` it to get a random permutation, keep the first `len_keep` indices, `gather` those rows out of the token tensor. A second `argsort` gives the inverse permutation used later to restore order. Because `len_keep` is a fixed count, every image in the batch keeps the same number of patches and the result is a single tensor of shape (N, len_keep, D).
6. **Asymmetric encoder/decoder.** The encoder only ever sees visible tokens, which is where the compute saving comes from. The decoder is narrower and shallower and sees all positions.
7. **Mask token.** A single learned vector broadcast into every masked slot before unshuffling.
8. **Reconstruction target and masked loss.** Target is the patchified pixels, optionally per-patch normalized (subtract patch mean, divide by patch std). MSE is averaged over the masked patches only.
9. **Weight transfer.** After pretraining, only the encoder is kept. The decoder, mask token, and reconstruction head are discarded.

### Shape trace, MAE ViT-B/16, N = batch size

| Step | Shape |
|---|---|
| input image | N, 3, 224, 224 |
| patchify target | N, 196, 768 |
| patch_embed | N, 196, 768 |
| + pos_embed (skip CLS slot) | N, 196, 768 |
| random_masking, mask_ratio 0.75 | N, 49, 768 |
| prepend CLS token | N, 50, 768 |
| 12 encoder blocks + norm | N, 50, 768 |
| decoder_embed | N, 50, 512 |
| drop CLS, append 147 mask tokens | N, 196, 512 |
| gather with ids_restore, re-attach CLS | N, 197, 512 |
| + decoder pos_embed, 8 decoder blocks | N, 197, 512 |
| decoder_pred | N, 197, 768 |
| drop CLS | N, 196, 768 |
| loss vs target, masked mean | scalar |

196 = (224/16)^2. 768 = 16 * 16 * 3, which coincidentally equals the encoder width.

### Concepts (ViTDet fine-tuning)

In order of execution. Steps that are the same for every detector are covered once, under path C.

1. **Weight loading.** Build the detector with random weights everywhere, then overwrite the backbone's weights with the ones from the MAE checkpoint, matched by parameter name. The CLS token is dropped, since detection has no use for a whole-image summary. The relative position bias parameters have no counterpart in the checkpoint and stay random.
2. **Large scale jitter.** Resize each image by a random factor between roughly 0.1x and 2.0x, then crop or pad to a fixed 1024 x 1024. The wide range of scales is what teaches one model to handle objects of very different sizes.
3. **Position embedding interpolation.** Pretraining ran at 224, so the position table covers a 14 x 14 grid. Detection runs at 1024, a 64 x 64 grid. The table is stretched to the new size by bilinear interpolation before training starts.
4. **Windowed attention with a few global blocks.** A block is one transformer layer: normalize, self-attention, add the input back, normalize, small MLP, add again. Self-attention compares every token to every other one, which is too costly at 4096 tokens. So most blocks cut the grid into separate, non-overlapping 14 x 14 tiles and attend only within each tile. Four evenly spaced blocks skip the tiling and attend across the whole grid.
5. **Plain, non-hierarchical backbone.** The ViT holds one resolution the whole way through, so its earlier blocks offer no coarser or finer views. Only the final block's output is used.
6. **Simple feature pyramid.** That one stride-16 map is resampled by fixed factors, 4x and 2x up, 1x, and 2x down, to produce the pyramid a ResNet would have produced from its stages.

### Shape trace, ViTDet-B detection

| Step | Shape |
|---|---|
| input image | N, 3, 1024, 1024 |
| patch_embed, stride 16 | N, 64, 64, 768 |
| + interpolated pos_embed | N, 64, 64, 768 |
| 12 blocks, windowed + global | N, 64, 64, 768 |
| permute to feature map | N, 768, 64, 64 |
| pyramid 4.0x -> p2 | N, 256, 256, 256 |
| pyramid 2.0x -> p3 | N, 256, 128, 128 |
| pyramid 1.0x -> p4 | N, 256, 64, 64 |
| pyramid 0.5x -> p5 | N, 256, 32, 32 |
| RPN per level | objectness + box deltas per anchor |
| ROI align on proposals | R, 256, 7, 7 |
| box head | R, 81 class logits and R, 320 box deltas |

R is the number of proposals fed to the box head, 512 per image during training.

## Path B: SparK pretraining -> Faster R-CNN fine-tuning

### Concepts (SparK pretraining)

In order of execution.

1. **Data loading.** Take a random region covering 67% to 100% of the image, resize it to 224 x 224, flip left-right at random, then subtract a fixed per-channel mean and divide by a fixed per-channel standard deviation. Both repos use the constants computed from ImageNet, which are standard regardless of the dataset being trained on. Labels are never read.
2. **Convolutional backbone.** A ResNet-50 processes the image in five steps, each halving the width and height and increasing the channel count. 224 becomes 112, 56, 28, 14, 7, so the total shrink factor is 32. The maps at 56, 28, 14, and 7 are the four stage outputs.
3. **Mask generation.** Decide the mask on the smallest grid, 7 x 7. Draw a random value for each of the 49 cells and keep the 40% with the lowest values, so 20 cells stay visible and 29 are hidden. Doing it here rather than at full resolution means one cell corresponds cleanly to one position in every stage.
4. **Applying the mask.** Scale the 7 x 7 mask up to 224 x 224 and zero out the hidden pixels of the input image. Scale it to each stage's size as well, since it must be reapplied inside the backbone.
5. **Sparse convolution.** A convolution reads a neighborhood, so at a mask boundary it mixes visible pixels with zeros and smears information into the hidden region. It also shifts the statistics the normalization layers see, because all those zeros are not real data. SparK prevents both: after every convolution the hidden positions are set back to zero, and the normalization layers compute their statistics over visible positions only.
6. **Hierarchical encoding.** Unlike a ViT, which holds one resolution throughout, the ResNet hands back four maps of different sizes. All four are kept and all four feed the decoder.
7. **Densify with mask tokens.** The four maps still have holes. At each scale, every hidden position is filled with a learned vector, one such vector per scale. After this the tensors are ordinary dense tensors and normal convolutions can run on them.
8. **Decoder.** Start with the smallest map, 7 x 7. Upsample it to 14 x 14, add in the encoder's 14 x 14 map, run a convolution. Repeat up to 28, then 56, then to full 224 x 224 with three output channels. Each round adds detail the previous round was too coarse to hold.
9. **Loss.** Cut both the reconstruction and the original image into patches. Within each patch of the original, subtract its mean and divide by its standard deviation, which removes brightness and contrast differences so the network is scored on structure. Take the squared difference, and average it over the hidden patches only.
10. **Weight transfer.** Keep the ResNet trunk. Throw away the decoder, the mask tokens, and everything else.

### Shape trace, SparK ResNet-50, 224 input, mask ratio 0.6

| Step | Shape |
|---|---|
| input image | N, 3, 224, 224 |
| mask grid at stride 32 | N, 1, 7, 7 |
| mask upsampled to input | N, 1, 224, 224 |
| masked input | N, 3, 224, 224 |
| stage 1 sparse features | N, 256, 56, 56 |
| stage 2 | N, 512, 28, 28 |
| stage 3 | N, 1024, 14, 14 |
| stage 4 | N, 2048, 7, 7 |
| densify each scale to decoder width | N, C_dec, 7, 7 ... N, C_dec, 56, 56 |
| decoder output | N, 3, 224, 224 |
| patchify both, masked L2 | scalar |

7 x 7 x (1 - 0.6) rounds to 20 visible cells out of 49.

### Concepts (Faster R-CNN fine-tuning)

In order of execution. Steps that are the same for every detector are covered once, under path C.

1. **Weight loading.** Build the detector with random weights everywhere, then overwrite the ResNet trunk with the SparK checkpoint. The decoder and mask tokens have no place in a detector and are discarded.
2. **Normalization swap.** SparK pretrains with normalization computed over visible positions only. Detection has no mask, so the backbone reverts to ordinary normalization, with the statistics frozen at their pretrained values by default.
3. **FPN.** The four ResNet stage outputs have different channel counts, so a 1x1 convolution brings each to 256. Then, starting from the smallest, each map is upsampled and added into the next larger one, so coarse context reaches the fine maps. That gives p2 through p5, and a max-pool of p5 gives p6.

Everything after this point, from anchors through evaluation, is identical to path A and is described under path C.

### Shape trace, Faster R-CNN R50-FPN, 800 x 1333 input

| Step | Shape |
|---|---|
| input image | N, 3, 800, 1333 |
| stage 2 (stride 4) | N, 256, 200, 336 |
| stage 3 | N, 512, 100, 168 |
| stage 4 | N, 1024, 50, 84 |
| stage 5 | N, 2048, 25, 42 |
| FPN p2 | N, 256, 200, 336 |
| FPN p3 | N, 256, 100, 168 |
| FPN p4 | N, 256, 50, 84 |
| FPN p5 | N, 256, 25, 42 |
| FPN p6 (maxpool) | N, 256, 13, 21 |
| ROI align | R, 256, 7, 7 |
| box head | R, 81 and R, 320 |

R is again the number of proposals fed to the box head.

## Path C: supervised training, no pretraining

This is the control arm. Same detectors and same evaluation, but the backbone starts from random weights instead of a pretrained checkpoint, and boxes are the only training signal. The schedule cannot be held identical: a random backbone needs a longer one to converge.

### Concepts

In order of execution. Everything here also happens during the fine-tuning runs in paths A and B; the only difference there is where the backbone weights came from.

1. **Data loading.** Read the image and its list of labeled boxes. Resize so the short side is one of six fixed values between 640 and 800, capping the long side at 1333. Flip left-right at random, moving the boxes with it. Subtract the channel means. Images in a batch have different sizes, so pad them all to a common size that is a multiple of 32, and record the real size of each so padding is never mistaken for image content.
2. **Backbone forward.** Push the batch through the ResNet or ViT and collect the feature maps. This is the part whose starting weights the experiment varies.
3. **Pyramid construction.** Turn those maps into five maps at 256 channels each, named p2 through p6, covering strides 4 through 64. A ResNet uses FPN: a 1x1 convolution on each stage to make the channel counts match, then add each coarse map into the finer one above it after upsampling. A ViT has only one map, so ViTDet resamples that single map to the five sizes instead.
4. **Anchors.** Lay a fixed set of reference boxes over every cell of every pyramid level. Detectron2's default is one size per level (32, 64, 128, 256, 512) times three aspect ratios (0.5, 1.0, 2.0), so three anchors per cell. These are not learned; they are the fixed guesses the network corrects.
5. **RPN head forward.** A small convolutional head runs on every pyramid level and outputs, per anchor, one objectness score and four numbers describing how to adjust that anchor's shape.
6. **Anchor matching.** Compare each anchor to each labeled box by intersection over union, the overlap area divided by the combined area. IoU at or above 0.7 makes the anchor positive, below 0.3 negative, in between is ignored and contributes nothing to the loss. Each labeled box also forces its single best anchor positive so no object goes unassigned.
7. **Box regression targets.** The network never predicts pixel coordinates. It predicts four adjustments that turn an anchor into the target box: how far to shift the center horizontally and vertically as a fraction of the anchor's width and height, and how much to scale the width and height, expressed as logarithms. Fractions and logarithms make the numbers independent of object size and keep them near zero, which trains more stably than raw pixels.
8. **RPN loss.** Sample 256 anchors per image, aiming for half positive. Binary cross entropy on their objectness, plus an L1 penalty on the four adjustments of the positive anchors only. Negative anchors have no box to match, so there is nothing to regress toward.
9. **Proposal generation.** Take the 2000 highest-scoring anchors, apply their predicted adjustments, clip the results to the image, drop degenerate ones, then run non-maximum suppression at IoU 0.7: keep the highest-scoring box, delete everything overlapping it past that threshold, repeat. Keep the top 1000 survivors.
10. **ROI sampling.** From those 1000, select 512 per image, aiming for 25% positive. A proposal is positive if it overlaps a labeled box by at least 0.5, and it takes on that box's class. The rest are labeled background.
11. **ROI align.** Each proposal is a rectangle of arbitrary size, but the box head needs a fixed input. Assign the proposal to the pyramid level matching its size, then sample a 7 x 7 grid of points inside it, reading each point with bilinear interpolation at its exact fractional coordinate. Rounding those coordinates to whole pixels, which the older ROI pooling did, misaligns the crop by up to half a cell and measurably costs accuracy.
12. **Box head forward.** Flatten each 7 x 7 crop, push it through two fully connected layers, and output one score per class plus background, and four adjustments per class.
13. **Box head loss.** Cross entropy over the class scores for all 512 sampled proposals, plus L1 on the four adjustments belonging to the correct class, for positive proposals only.
14. **Summing and optimizing.** Add the four losses with equal weight, backpropagate through the heads, the pyramid, and the backbone alike, and step the optimizer. Nothing is frozen except, by default, the first ResNet stage.
15. **Inference.** No sampling and no loss. Apply the predicted adjustments, discard anything scoring under 0.05, run non-maximum suppression at 0.5 separately within each class, and keep the 100 highest-scoring detections per image.
16. **Evaluation.** COCO average precision. For each class, sort detections by score and walk down the list counting how many are correct at a given IoU threshold, which traces out a precision-recall curve whose area is the average precision. Repeat at thresholds 0.5 through 0.95 in steps of 0.05 and average everything. A detection counts as correct only if it overlaps an unclaimed labeled box by at least the threshold.
17. **Training from scratch caveats.** A pretrained backbone ships with frozen BatchNorm statistics inherited from its checkpoint, which a random backbone does not have. Group normalization or synchronized BatchNorm replaces it. A random backbone also needs a much longer schedule, several times the 90k iterations that suffice for a pretrained one. This is why the from-scratch control cannot simply reuse the pretrained arm's schedule.

### Standard supervised recipe, Detectron2 1x

| Setting | Value |
|---|---|
| Batch size | 16 images |
| Base LR | 0.02, SGD with momentum 0.9 |
| Iterations | 90000, LR divided by 10 at 60000 and 80000 |
| Train short side | one of 640, 672, 704, 736, 768, 800 |
| Test short side | 800, long side capped at 1333 |
| Anchors | 3 per cell across p2 to p6 |
| RPN sample | 256 anchors per image |
| ROI sample | 512 proposals per image, 25% positive |

This recipe is the Faster R-CNN one. ViTDet does not use it: it trains with AdamW instead of SGD, a layer-wise learning rate decay down the backbone, large scale jitter at 1024 x 1024, and roughly 100 epochs rather than 90k iterations. Each detector must therefore be compared against its own control, never across the two.

The forward shapes are identical to the fine-tuning traces above. Only the initialization and the schedule length differ.

### Losses in the supervised arm

| Loss | Applies to | Form |
|---|---|---|
| RPN objectness | sampled anchors | binary cross entropy |
| RPN box | positive anchors | smooth L1 on deltas |
| ROI classification | sampled proposals | cross entropy over the classes plus background |
| ROI box | positive proposals | smooth L1 on deltas |

All four are summed with equal weight.

## The comparison this project runs

| Arm | Backbone init | Labeled data |
|---|---|---|
| A1 | MAE pretrained ViT-B | full COCO |
| A2 | random ViT-B | full COCO |
| B1 | SparK pretrained R50 | full COCO |
| B2 | random R50 | full COCO |

Each arm repeats at reduced label fractions to measure where pretraining helps most.

## What is shared across both paths

- Mask a large fraction of the input, reconstruct the hidden pixels, and score the loss on the hidden part only.
- Learned mask token standing in for every removed position.
- Per-patch normalized pixel target.
- Encoder is the deliverable; everything else is scaffolding thrown away before fine-tuning.

## What differs

| | MAE | SparK |
|---|---|---|
| Backbone | plain ViT | hierarchical CNN |
| Masked units are | dropped from the sequence | zeroed in place |
| Encoder sees | 25% of tokens | full-size tensor with holes |
| Mask ratio | 0.75 | 0.6 |
| Compute saving | yes, shorter sequence | no, sparse ops on full resolution |
| Decoder | works at one zoom level throughout, guessing the pixels of each 16x16 block | starts from a coarse blurry image and repeatedly doubles its size, mixing in a sharper version from the encoder at each step until back to full resolution |
