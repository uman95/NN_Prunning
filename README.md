📖 Documentation 
================
### Refrence
https://github.com/VainF/Torch-Pruning

### How to Run this project

Install the requirements:
```bash
pip3 install -r requirements.txt 
```

To Run a baseline model for channel==3 i.e RGB
```bash
!python3 main_train.py --model 'ResNet34' --num_channel 1 --cuda --gpuids 0 --epochs 1 --data cifar10
```
To run l1_norm
```bash
! python3 main_train.py --model 'VGG16' --num_channel 1 --model_path 'l1_norm/model/VGG16/grey' --gpuids 0 --epochs 20 --data cifar10 
run vggprune
!python l1_norm/vggprune.py --num_channel 1 --dataset cifar10 --model l1_norm/model/VGG16/grey/chpt_est.pth --save l1_norm/prune/vgg/grey
```
