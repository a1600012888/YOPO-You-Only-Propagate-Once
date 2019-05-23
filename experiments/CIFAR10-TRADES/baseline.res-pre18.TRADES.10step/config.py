import sys
import os
import argparse
import numpy as np
import torch


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(lib_dir)

from training.config import TrainingConfigBase, SGDOptimizerMaker, \
    PieceWiseConstantLrSchedulerMaker, IPGDAttackMethodMaker

class TrainingConfing(TrainingConfigBase):

    create_optimizer = None #SGDOptimizerMaker(lr =1e-1 * 5 / K, momentum = 0.9, weight_decay = 5e-4)
    create_lr_scheduler = None #PieceWiseConstantLrSchedulerMaker(milestones = [35, 40, 45], gamma = 0.1)
    #
    create_loss_function = None #torch.nn.CrossEntropyLoss
    create_attack_method = None

    create_evaluation_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 20, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))


config = TrainingConfing()


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=1.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('-d', default=0, type=int, help='which gpu to use')

args = parser.parse_args()

if __name__ == '__main__':
    pass