from config import config
from dataset import create_test_dataset
from network import create_network

from training.train import eval_one_epoch
from utils.misc import load_checkpoint

import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '--resume', default='log/models/last.checkpoint',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default:log/last.checkpoint)')
parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
args = parser.parse_args()


DEVICE = torch.device('cuda:{}'.format(args.d))
torch.backends.cudnn.benchmark = True

net = create_network()
net.to(DEVICE)

ds_val = create_test_dataset(512)

AttackMethod = config.create_evaluation_attack_method(DEVICE)

if os.path.isfile(args.resume):
    load_checkpoint(args.resume, net)


print('Evaluating')
clean_acc, adv_acc = eval_one_epoch(net, ds_val, DEVICE, AttackMethod)
print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))
