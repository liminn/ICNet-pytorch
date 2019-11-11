# Description
This repo contains ICNet implemented by PyTorch, based on [paper](https://arxiv.org/abs/1704.08545) by Hengshuang Zhao, and et. al(ECCV'18).
Training and evaluation are done on the [Cityscapes dataset](https://www.cityscapes-dataset.com/) by default.

# Requirements
Python 3.6 or later with the following `pip3 install -r requirements.txt`:
- torch==1.1.0
- torchsummary==1.5.1
- torchvision==0.3.0
- numpy==1.17.0
- Pillow==6.0.0
- PyYAML==5.1.2

# Performance  
| Method | mIoU(%)  | Time(ms) | FPS | Memory(GB)| GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ICNet(paper)  | **67.7%**  | 33ms | 30.3 | **1.6** | TitanX|
| ICNet(ours)  | 66.7%  | **19ms** | **52.6** | 1.86    | GTX 1080Ti|
- A brief description of the experiment is as follows: only train on trainning set of Cityscapes, and test on validation set of Cityscapes, using only one GTX 1080Ti card, and input size of the test phase is $$2048x1024x3$$.
- For the performance of the original model, you can query the "Table2" in the [paper](https://arxiv.org/abs/1704.08545). 
- I have uploaded my pretrained models: `ckpt/icnet_resnet50_182_0.667_best_model.pth`

# Demo
|input|output|
|:---:|:---:|
|![src](https://github.com/liminn/ICNet/raw/master/demo/frankfurt_000001_057181_leftImg8bit_src.png)|![predict](https://github.com/liminn/ICNet/raw/master/demo/frankfurt_000001_057181_leftImg8bit_mIoU_0.680.png)|
|![src](https://github.com/liminn/ICNet/raw/master/demo/lindau_000005_000019_leftImg8bit_src.png)|![predict](https://github.com/liminn/ICNet/raw/master/demo/lindau_000005_000019_leftImg8bit_mIoU_0.657.png) |
|![src](https://github.com/liminn/ICNet/raw/master/demo/munster_000075_000019_leftImg8bit_src.png)|![predict](https://github.com/liminn/ICNet/raw/master/demo/munster_000075_000019_leftImg8bit_mIoU_0.672.png) |
- All the input images comes from the validation dataset of the Cityscaps.

# Usage
## Trainning
First, modify the configuration in the `configs/icnet.yaml` file:
```Python
### 3.Trainning 
train:
  specific_gpu_num: "1"        # for example: "0", "1" or "0, 1"
  train_batch_size: 25         # adjust according to gpu resources
  cityscapes_root: "/home/datalab/ex_disk1/open_dataset/Cityscapes/" 
  ckpt_dir: "./ckpt/" # ckpt and trainning log will be saved here
```
Then, run: `python3 train.py`

## Evaluation
First, modify the configuration in the `configs/icnet.yaml` file:
```Python
### 4.Test
test:
  ckpt_path: "./ckpt/icnet_resnet50_182_0.667_best_model.pth"  # set the pretrained model path correctly
```
Then, run: `python3 evaluate.py`

# Discussion
Issues should be raised directly in the repository. I will be replay as soon as possible.

# Reference
- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)