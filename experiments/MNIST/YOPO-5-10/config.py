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

    lib_dir = lib_dir

    num_epochs = 40
    val_interval = 1
    weight_decay = 5e-4

    inner_iters = 10
    K = 5
    sigma = 0.01
    eps = 0.3

    create_optimizer = SGDOptimizerMaker(lr =1e-1 / K, momentum = 0.9, weight_decay = weight_decay)
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [30, 35, 39], gamma = 0.1)

    create_loss_function = None

    create_attack_method = None

    create_evaluation_attack_method = \
        IPGDAttackMethodMaker(eps = 0.3, sigma = 0.01, nb_iters = 40, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))


config = TrainingConfing()


parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                 help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                 metavar='N', help='mini-batch size')
parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
parser.add_argument('-adv_coef', default=1.0, type = float,
                    help = 'Specify the weight for adversarial loss')
parser.add_argument('--auto-continue', default=False, action = 'store_true',
                    help = 'Continue from the latest checkpoint')
args = parser.parse_args()


if __name__ == '__main__':
    pass
