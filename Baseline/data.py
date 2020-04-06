import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from .config import cfg


# TODO: Edit this file to cater for CIFAR10 and ImageNet

# Download & Load Dataset


print('==> Preparing data..')

opt = cfg

if opt.num_channel == 3:

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.47336,), (0.2507,)),
#         transforms.Lambda(lambda x: 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]),
#         transforms.Lambda(lambda x: x.unsqueeze(dim=0)),
    ])
    
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.47336,), (0.2507,)),
#         transforms.Lambda(lambda x: 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]),
#         transforms.Lambda(lambda x: x.unsqueeze(dim=0)),
    ])



def get_dataloader(opt):

    if opt.data =='cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
         transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
         transform=transform_val)

    elif opt.data == 'imagenet':
        train_dataset = datasets.ImageNet(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = datasets.ImageNet(root='./data', train=False, download=True, transform=transform_train)
    else:
        print(" Enter a valid dataset")

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=1)#opt.workers)

    test_loader = DataLoader(val_dataset,
                             batch_size=opt.test_batch_size,
                             shuffle=False,
                             num_workers=1)#opt.workers)

    return train_loader , test_loader



# train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.47336,), (0.2507,))]), download=True),
#                                            shuffle=True, batch_size=32, num_workers=0)
#
# test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.47336,), (0.2507,))]), download=True),
#                                           shuffle=True, batch_size=32, num_workers=0)



