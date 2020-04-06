from .models.mobilenet import MobileNet
from .models.resnet import ResNet34, ResNet50
from l1_norm.configs.prune_config import args
from .models.vgg import VGG


def Model():
    opt = args
    if opt.model == 'MobileNet':
        return MobileNet(opt.num_channel)
    elif opt.model == 'ResNet34':
        return ResNet34(opt.num_channel)
    elif opt.model == 'ResNet50':
        return ResNet50(opt.num_channel)
    elif opt.model == 'VGG16':
        return VGG(opt.num_channel, 'VGG16')
    else:
        return print("Please enter a valid Model") #VGG(opt.num_channel)