## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

MiDaS v2.1 was trained on 10 datasets (ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir, HRWSI, ApolloScape, BlendedMVS, IRS) with
multi-objective optimization. 
The original model that was trained on 5 datasets  (`MIX 5` in the paper) can be found [here](https://github.com/intel-isl/MiDaS/releases/tag/v2).

1) Download the model weights [model-f6b98070.pt](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt) 
and [model-small-70d6b9c8.pt](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt) and place the
file in the root folder.
  
#### via PyTorch Hub

The pretrained model is also available on [PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)

### Accuracy

Zero-shot error (the lower - the better) and speed (FPS):

| Model |  DIW, WHDR | Eth3d, AbsRel | Sintel, AbsRel | Kitti, δ>1.25 | NyuDepthV2, δ>1.25 | TUM, δ>1.25 | Speed, FPS |
|---|---|---|---|---|---|---|---|
| **Small models:** | | | | | | | iPhone 11 |
| MiDaS v2 small | **0.1248** | 0.1550 | **0.3300** | **21.81** | 15.73 | 17.00 | 0.6 |
| MiDaS v2.1 small [URL](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt) | 0.1344 | **0.1344** | 0.3370 | 29.27 | **13.43** | **14.53** | 30 |
| Relative improvement | -7.7% | **+13.3%** | -2.1% | -34.2% | **+14.6%** | **+14.5%** | **50x** |
| | | | | | | |
| **Big models:** | | | | | | | GPU RTX 2080Ti |
| MiDaS v2 large [URL](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt) | **0.1246** | 0.1290 | **0.3270** | 23.90 | 9.55 | 14.29 | 59 |
| MiDaS v2.1 large [URL](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt) | 0.1295 | **0.1155** | 0.3285 | **16.08** | **8.71** | **12.51** | 59 |
| Relative improvement | -3.9% | **+10.5%** | -0.52% | **+32.7%** | **+8.8%** | **+12.5%** | 1x |

