import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 

# TODO: Edit this file to cater for CIFAR10 and ImageNet


def get_dataloader():
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data',train=True, transform= transforms.Compose([
                          transforms.Grayscale(num_output_channels=1),
                          transforms.ToTensor(),
                          transforms.Normalize((0.47336,), (0.2507,))]),download = True),
                          shuffle=True, batch_size=32, num_workers = 0)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data',train=False, transform= transforms.Compose([
                          transforms.Grayscale(num_output_channels=1),
                          transforms.ToTensor(),
                          transforms.Normalize((0.47336,), (0.2507,))]),download = True),
                          shuffle=True, batch_size=32, num_workers = 0)

    return train_loader , test_loader

