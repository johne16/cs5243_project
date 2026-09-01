# Vision + Depth Fusion

## The idea

One option is benchmarking radar + camera fusion models that predict dense depth from an RGB image plus a sparse radar point cloud. Camera gives detail but no true scale, radar gives accurate range at a handful of points. The comparison would be how different fusion strategies combine them.

This ties into your robot vision idea. The datasets are from self-driving cars, but drones and ground robots have the same problem.

## Data

**View-of-Delft** looks like a good fit. 8600 frames, 3+1D radar (richer than nuScenes radar), 64-beam lidar for dense depth supervision, and a stereo camera, which would let us compare against a second-camera baseline rather than only monocular. Access needs an institutional email and manual approval from TU Delft, so it may be worth requesting early either way.
https://viewofdelft-dataset.tudelft.nl/

**nuScenes** could serve as the fallback, or as the place to build the pipeline first. Instant access after registration, and every model below already targets it.
https://www.nuscenes.org/nuscenes

Neither has dense depth labels. Ground truth comes from projecting lidar into the camera frame and computing loss only at those pixels. Lidar supervises, camera and radar are the only inputs at test time.

## Models to compare

| Model | Approach |
|---|---|
| [Radar-Camera Depth](https://github.com/nesl/radar-camera-fusion-depth) (CVPR 2023) | Maps each radar point to possible image surfaces |
| [RADIANT](https://github.com/longyunf/radiant) (AAAI 2023) | Radar corrects a monocular 3D detector |
| [RCDPT](https://github.com/lochenchou/rcdpt) (ICASSP 2023) | Transformer, fuses inside a dense prediction network |

Possible baselines: monocular with no radar, to isolate what radar adds. On VoD, stereo as well, to ask whether radar buys anything a second camera does not.

## What we might add

- Reproduce all three under one evaluation protocol. The papers use different splits and preprocessing.
- Break error down by distance from sensor and radar point density.
