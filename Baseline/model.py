from .models.mobilenet import MobileNet
from .models.vgg import VGG
from .models.resnet import ResNet18, ResNet50
from .config import cfg


def Model():
    opt = cfg
    if opt.model == 'MobileNet':
        return MobileNet(opt.num_channel)
    elif opt.model == 'ResNet18':
        return ResNet18(opt.num_channel)
    elif opt.model == 'ResNet50':
        return ResNet50(opt.num_channel)
    elif opt.model == 'VGG16':
        return VGG(opt.num_channel, 'VGG16')
    else:
        return print("Please enter a valid Model") #VGG(opt.num_channel)