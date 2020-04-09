# Network Slimming

This directory contains the pytorch implementation for [network slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  

## Channel Selection Layer
We introduce `channel selection` layer to help the  pruning of ResNet and DenseNet. This layer is easy to implement. It stores a parameter `indexes` which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.

To run any of the script below you need to chenge directory to the slimming by:
``` shell
cd slimming/
```

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg`,`resnet` or
`densenet`. The depth is chosen to be the same as the networks used in the paper. The 3 commands to run are these.
```shell
python3 main.py --dataset cifar10 --num_channel 3 (or 1) --save ./logs/vgg/rgb (grey) --arch vgg --depth 19
python3 main.py --dataset cifar10 --num_channel 3 (or 1) --save ./logs/resnet/rgb (grey) --arch resnet --depth 164
python3 main.py --dataset cifar10 --num_channel 3 (or 1) --save ./logs/densenet/rgb (grey) --arch densenet --depth 40
```

## Train with Sparsity
```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40
```

## Prune

```shell
python3 vggprune.py --dataset cifar10 --num_channel 3 (1) --depth 19 --percent 0.7 --model logs/vgg/rgb (grey) --save pruned_models/vgg/rgb (grey)
python3 resprune.py --dataset cifar10 --num_channel 3 --depth 164 --percent 0.4 --model logs/resnet/rgb (grey) --save pruned_models/resnet/rgb (grey)
python3 denseprune.py --dataset cifar10 --num_channel 3 --depth 40 --percent 0.4 --model logs/resnet/rgb (grey) --save pruned_models/densenet/rgb (grey)
```
The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 164
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch densenet --depth 40
```

## Scratch-E
```
python main_E.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19
python main_E.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 164
python main_E.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch densenet --depth 40
```

## Scratch-B
```
python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19
python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 164
python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch densenet --depth 40
```

