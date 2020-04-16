# Pruning Filters for Efficient Convnets ImageNet

We get the ResNet-34 baseline model from Pytorch model zoo.


## Baseline
We trained the Baseline models again
```
python3 main.py -a resnet34 --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Prune
```
python prune.py -v A --save [PATH TO SAVE RESULTS] [IMAGENET]

```

