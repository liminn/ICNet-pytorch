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
- Base on Cityscapes dataset, only train on trainning set, and test on validation set, using only one GTX 1080Ti card, and input size of the test phase is 2048x1024x3.
- For the performance of the original paper, you can query the "Table2" in the [paper](https://arxiv.org/abs/1704.08545). 
- I have uploaded my pretrained models: `ckpt/icnet_resnet50_182_0.667_best_model.pth`

# Demo
|input|output|
|:---:|:---:|
|![src](https://github.com/liminn/ICNet/raw/master/demo/frankfurt_000001_057181_leftImg8bit_src.png)|![predict](https://github.com/liminn/ICNet/raw/master/demo/frankfurt_000001_057181_leftImg8bit_mIoU_0.680.png)|
|![src](https://github.com/liminn/ICNet/raw/master/demo/lindau_000005_000019_leftImg8bit_src.png)|![predict](https://github.com/liminn/ICNet/raw/master/demo/lindau_000005_000019_leftImg8bit_mIoU_0.657.png) |
|![src](https://github.com/liminn/ICNet/raw/master/demo/munster_000075_000019_leftImg8bit_src.png)|![predict](https://github.com/liminn/ICNet/raw/master/demo/munster_000075_000019_leftImg8bit_mIoU_0.672.png) |
- All the input images comes from the validation dataset of the Cityscaps, you can switch to the `demo/` directory to check more demo results.

# Usage
## Trainning
First, modify the configuration in the `configs/icnet.yaml` file:
```Python
### 3.Trainning 
train:
  specific_gpu_num: "1"   # for example: "0", "1" or "0, 1"
  train_batch_size: 25    # adjust according to gpu resources
  cityscapes_root: "/home/datalab/ex_disk1/open_dataset/Cityscapes/" 
  ckpt_dir: "./ckpt/"     # ckpt and trainning log will be saved here
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
![ICNet](https://github.com/liminn/ICNet/raw/master/ICNet.png)
The structure of ICNet is mainly composed of `sub4`, `sub2`, `sub1` and `head`: 
- `sub4`: basically a `pspnet`, the most difference is a modified `pyramid pooling module`.
- `sub2`: the first three stages of `sub4`, `sub2` and `sub4` share these three phases convolutional layers.
- `sub1`: three consecutive stried convolutional layers, to fastly downsample the original large-size input images
- `head`: through the `CFF` module, the outputs of the three cascaded branches( `sub4`, `sub2` and `sub1`) are connected. Finaly, using 1x1 convolution and interpolation to get the output.

During the training, I found that `pyramid pooling module` in `sub4` is very important. It can significantly improve the performance of the network and lightweight models. 
In addition, I found that a small training technique can improve the performance of the model: 
- set the learning rate of `sub4` to orginal initial learning rate(0.01), because it has backbone pretrained weights.
- set the learning rate of `sub1` and `head` to 10 times initial learning rate(0.1), because there are no pretrained weights for them.

This small training technique is really effective, it can improve the mIoU performance by 1~2 percentage points.

Any other questions or my mistakes can be fedback in the comments section. I will replay as soon as possible.

# Reference
- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)