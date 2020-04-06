import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import datasets, transforms

from Baseline.models.resnet import ResNet34
from utils import validate
from l1_norm.prune_config import args


args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)


model = ResNet34(in_channel=3)
model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True

print('Pre-processing Successful!')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(os.path.join(args.data,'val'), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()

skip = {
    'A': [2, 8, 14, 16, 26, 28, 30, 32],
    'B': [2, 8, 14, 16, 26, 28, 30, 32],
}

prune_prob = {
    'A': [0.3, 0.3, 0.3, 0.0],
    'B': [0.5, 0.6, 0.4, 0.0],
}


layer_id = 1
cfg = []
cfg_mask = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1,1):
            continue
        out_channels = m.weight.data.shape[0]
        if layer_id in skip[args.v]:
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
            layer_id += 1
            continue
        if layer_id % 2 == 0:
            if layer_id <= 6:
                stage = 0
            elif layer_id <= 14:
                stage = 1
            elif layer_id <= 26:
                stage = 2
            else:
                stage = 3
            prune_prob_stage = prune_prob[args.v][stage]
            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1,2,3))
            num_keep = int(out_channels * (1 - prune_prob_stage))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            cfg.append(num_keep)
            layer_id += 1
            continue
        layer_id += 1

assert len(cfg) == 16, "Length of cfg variable is not correct."

newmodel = ResNet34(cfg=cfg)
newmodel = torch.nn.DataParallel(newmodel).cuda()

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        if m0.kernel_size == (1,1):
            # Cases for down-sampling convolution.
            m1.weight.data = m0.weight.data.clone()
            continue
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue
    elif isinstance(m0, nn.BatchNorm2d):
        assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))


acc_top1, acc_top5 = validate(val_loader, model, criterion, print_freq=args.print_freq)
new_acc_top1, new_acc_top5 = validate(val_loader, newmodel, criterion, print_freq=args.print_freq)
num_parameters1 = sum([param.nelement() for param in model.parameters()])
num_parameters2 = sum([param.nelement() for param in newmodel.parameters()])

# TODO: Check if saving report saves for each model or it overwrites the file.

with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Before pruning: "+"\n")
    fp.write("acc@1: "+str(acc_top1)+"\n"+"acc@5: "+str(acc_top5)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters1)+"\n")
    fp.write("==========================================\n")
    fp.write("After pruning: "+"\n")
    fp.write("cfg :"+"\n")
    fp.write(str(cfg)+"\n")
    fp.write("acc@1: "+str(new_acc_top1)+"\n"+"acc@5: "+str(new_acc_top5)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters2)+"\n")