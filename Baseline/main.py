from __future__ import print_function
import time

from os.path import isfile, join

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from Baseline.data import get_dataloader
from utils import adjust_learning_rate, train, validate, save_model
from .config import cfg
from .model import Model


best_prec1 = 0


def main():
    global opt, start_epoch, best_prec1
    opt = cfg
    opt.gpuids = list(map(int, opt.gpuids))
    train_loader, val_loader = get_dataloader(opt)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=True)
    start_epoch = 0

    ckpt_file = join("model", opt.ckpt)

    # Running o
    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids,
                                output_device=opt.gpuids[0])
        cudnn.benchmark = True

    # for resuming training
    if opt.resume:
        if isfile(ckpt_file):
            print("==> Loading Checkpoint '{}'".format(opt.ckpt))
            if opt.cuda:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(opt.gpuids[0]))
                try:
                    model.module.load_state_dict(checkpoint['model'])
                except:
                    model.load_state_dict(checkpoint['model'])
            else:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
                try:
                    model.load_state_dict(checkpoint['model'])
                except:
                    # create new OrderedDict that does not contain `module.`
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model'].items():
                        if k[:7] == 'module.':
                            name = k[7:] # remove `module.`
                        else:
                            name = k[:]
                        new_state_dict[name] = v

                    model.load_state_dict(new_state_dict)
            
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("==> Loaded Checkpoint '{}' (epoch {})".format(opt.ckpt, start_epoch))
        else:
            print("==> no checkpoint found at '{}'".format(opt.ckpt))
            return

    # for evaluation
    if opt.eval:
        if isfile(ckpt_file):
            print("==> Loading Checkpoint '{}'".format(opt.ckpt))
            if opt.cuda:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(opt.gpuids[0]))
                try:
                    model.module.load_state_dict(checkpoint['model'])
                except:
                    model.load_state_dict(checkpoint['model'])
            else:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
                try:
                    model.load_state_dict(checkpoint['model'])
                except:
                    # create new OrderedDict that does not contain `module.`
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model'].items():
                        if k[:7] == 'module.':
                            name = k[7:] # remove `module.`
                        else:
                            name = k[:]
                        new_state_dict[name] = v

                    model.load_state_dict(new_state_dict)
            
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("==> Loaded Checkpoint '{}' (epoch {})".format(opt.ckpt, start_epoch))

            # evaluate on validation set
            print("\n===> [ Evaluation ]")
            start_time = time.time()
            prec1 = validate(val_loader, model, criterion, opt)

            # TODO: what happens to prec1
            # TODO: Tensoroard logging
            elapsed_time = time.time() - start_time
            print("====> {:.2f} seconds to evaluate this model\n".format(elapsed_time))
            return
        else:
            print("==> no checkpoint found at '{}'".format(
                        opt.ckpt))
            return

    # train...
    train_time = 0.0
    validate_time = 0.0
    for epoch in range(start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch, opt.lr)

        print('\n==> Epoch: {}, lr = {}'.format(epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        print("===> [ Training ]")
        start_time = time.time()
        train(train_loader, model, criterion, optimizer, opt)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print("====> {:.2f} seconds to train this epoch\n".format(elapsed_time))
        
        # evaluate on validation set
        print("===> [ Validation ]")
        start_time = time.time()
        prec1 = validate(val_loader, model, criterion, print_freq=10)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print("====> {:.2f} seconds to validate this epoch\n".format(elapsed_time))

        # remember best prec@1 and save checkpoint
        print(best_prec1,prec1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {'best_prec1':best_prec1,'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_model(opt.model_path,state, epoch, is_best)
    
    avg_train_time = train_time/opt.epochs
    avg_valid_time = validate_time/opt.epochs
    total_train_time = train_time+validate_time
    print("====> average training time per epoch: {}m {:.2f}s".format(int(avg_train_time//60), avg_train_time%60))
    print("====> average validation time per epoch: {}m {:.2f}s".format(int(avg_valid_time//60), avg_valid_time%60))
    print("====> training time: {}m {:.2f}s".format(int(train_time//60), train_time%60))
    print("====> validation time: {}m {:.2f}s".format(int(validate_time//60), validate_time%60))
    print("====> total training time: {}m {:.2f}s".format(int(total_train_time//60), total_train_time%60))

