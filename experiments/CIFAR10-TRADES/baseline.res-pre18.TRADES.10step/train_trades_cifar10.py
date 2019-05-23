from __future__ import print_function
import os
from tqdm import tqdm
from collections import OrderedDict
from time import time
import json

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from config import config, args
from network import create_network
from trades import trades_loss

from training.train import eval_one_epoch
from utils.misc import torch_accuracy, AvgMeter

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda:{}'.format(args.d) if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch, descrip_str='Training'):
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description(descrip_str)

    CleanAccMeter = AvgMeter()
    TradesAccMeter = AvgMeter()
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, cleanloss, klloss, cleanacc, tradesacc = trades_loss(model=model,
                                                                   x_natural=data,
                                                                   y=target,
                                                                   optimizer=optimizer,
                                                                   device=device,
                                                                   step_size=args.step_size,
                                                                   epsilon=args.epsilon,
                                                                   perturb_steps=args.num_steps,
                                                                   beta=args.beta,)
        loss.backward()
        optimizer.step()

        CleanAccMeter.update(cleanacc)
        TradesAccMeter.update(tradesacc)

        pbar_dic = OrderedDict()
        pbar_dic['cleanloss'] = '{:.3f}'.format(cleanloss)
        pbar_dic['klloss'] = '{:.3f}'.format(klloss)
        pbar_dic['CleanAcc'] = '{:.2f}'.format(CleanAccMeter.mean)
        pbar_dic['TradesAcc'] = '{:.2f}'.format(TradesAccMeter.mean)
        pbar.set_postfix(pbar_dic)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    model = create_network().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    EvalAttack = config.create_evaluation_attack_method(device)

    now_train_time = 0
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        s_time = time()
        descrip_str = 'Training epoch: {}/{}'.format(epoch, args.epochs)
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, descrip_str)
        now_train_time += time() - s_time

        acc, advacc = eval_one_epoch(model, test_loader, device, EvalAttack)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(config.model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))


if __name__ == '__main__':
    main()