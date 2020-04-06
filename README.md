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
#python3 main_train.py --model 'ResNet34' --num_channel 3 --epochs 1 --data cifar10
!python3 main_train.py --model 'ResNet34' --num_channel 1 --cuda --gpuids 0 --epochs 1 --data cifar10
```
