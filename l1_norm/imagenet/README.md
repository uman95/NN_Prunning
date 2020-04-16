# Pruning Filters for Efficient Convnets ImageNet

We get the ResNet-34 baseline model from Pytorch model zoo.


## Baseline
We trained the Baseline models again
```
RGB

python3 main.py -a resnet34 --num_channel 3 --save 'ImagNet/model/Resnet34/RGB' [IMAGENET]

Greay

python3 main.py -a resnet34 --num_channel 1 --save 'ImagNet/model/Resnet34/Greay'  [IMAGENET]
```

## Prune
```
python prune.py -v A --save [PATH TO SAVE RESULTS] [IMAGENET]

```

