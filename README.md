📖 Documentation 
================
### Refrence
https://github.com/VainF/Torch-Pruning

### How to Run this project

Install the requirements:
```bash
pip3 install -r requirements.txt 
```
FOR CIFAR DATASET

To Run a baseline model for channel==3 

RGB
```bash

!python3 main_train.py --model 'ResNet50' --num_channel 3 --model_path 'l1_norm/model/ResNet50/RGB' --cuda --gpuids 0 1 --epochs 150 --data cifar10 
!python3 main_train.py --model 'VGG16' --num_channel 3 --model_path 'l1_norm/model/VGG16/RGB' --cuda --gpuids 0 1 --epochs 150 --data cifar10 

```
GREY
```bash
! python3 main_train.py --model 'ResNet50' --num_channel 1 --model_path 'l1_norm/model/ResNet50/grey' --cuda --gpuids 0 1 --epochs 150 --data cifar10 
! python3 main_train.py --model 'VGG16' --num_channel 1 --model_path 'l1_norm/model/VGG16/grey' --cuda --gpuids 0 1 --epochs 150 --data cifar10 
```
Run Prune l1_norm 

RGB
```bash
!python l1_norm/resnetprune.py --num_channel 3 --dataset cifar10 --model l1_norm/model/resnet/RGB/ckpt_best.pth --save l1_norm/prune/resnset/RGB
!python l1_norm/vggprune.py --num_channel 3 --dataset cifar10 --model l1_norm/model/VGG16/RGB/ckpt_best.pth --save l1_norm/prune/vgg/RGB

```

GREY
```bash
!python l1_norm/resnetprune.py --num_channel 1 --dataset cifar10 --model l1_norm/model/resnet/grey/ckpt_best.pth --save l1_norm/prune/resnset/grey
!python l1_norm/vggprune.py --num_channel 1 --dataset cifar10 --model l1_norm/model/VGG16/grey/ckpt_best.pth --save l1_norm/prune/vgg/grey

```
