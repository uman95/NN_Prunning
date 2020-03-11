from mobilenet import MobileNet
from vgg import VGG
from resnet import ResNet18, ResNet50
from config import cfg


def Model():
    opt = cfg
    if opt.model == 'MobileNet':
        return MobileNet(opt.num_channel)
    elif opt.model == 'ResNet18':
        return ResNet18(opt.num_channel)
    elif opt.model == 'ResNet50':
        return ResNet50(opt.num_channel)
    else:
        return VGG(opt.num_channel)