
# Introduction
This directory contains PyTorch ICNet, based on [paper](https://arxiv.org/abs/1704.08545) by Hengshuang Zhao, and et. al(ECCV'18).

# Description
The https://github.com/ultralytics/yolov3 repo contains training and evaluate code for ICNet in PyTorch. Training and evaluate is done on the [Cityscapes dataset](https://www.cityscapes-dataset.com/) by default.

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

# Performance  

| Method(*) | mIoU(%)  | Time(ms) | FPS | Memory(GB)| GPU |
|:-:---|:-:---|:-:---|:-:---|:-:---|:-:---|
| ICNet(paper)  | **67.7%**  | 33ms | 30.3 | **1.6** | TitanX
| ICNet(ours)  | 66.7%  | **19ms** | **52.6** | 1.86    | GTX 1080Ti
*: 
- Input size: $2048x1024x3$, only train on trainning set of Cityscapes, and test on validation set of Cityscapes, using only one GTX 1080Ti card.
- pretrained models link: [icnet_resnet50_182_0.667_best_model.pth]()  

# Demo
|src|predict|
|:-:---|:-:---|
|./demo/frankfurt_000001_057181_leftImg8bit_src.png|./demo/frankfurt_000001_057181_leftImg8bit_mIoU_0.680.png|

|./demo/lindau_000005_000019_leftImg8bit_src.png|./demo/lindau_000005_000019_leftImg8bit_mIoU_0.657.png |

|./demo/munster_000106_000019_leftImg8bit_src.png|./demo/munster_000106_000019_leftImg8bit_mIoU_0.672.png|
|./demo/munster_000158_000019_leftImg8bit_src.png|./demo/munster_000158_000019_leftImg8bit_mIoU_0.658.png|

# Usage
## Config

## Train
`python3 train.py`
## evaluate
`python3 evaluate.py`

# Reference
- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)

# Contact
Issues should be raised directly in the repository. I will be replay as soon as possible.
