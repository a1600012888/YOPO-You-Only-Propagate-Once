from abc import ABCMeta, abstractproperty, abstractmethod
from typing import Tuple, List, Dict
import os
import sys
import torch


class TrainingConfigBase(metaclass=ABCMeta):
    '''
    Base class for training
    '''

    # directory handling
    @property
    def abs_current_dir(self):
        return os.path.realpath('./')

    @property
    def log_dir(self):
        if not os.path.exists('./log'):
            os.mkdir('./log')
        return os.path.join(self.abs_current_dir, 'log')

    @property
    def model_dir(self):
        log_dir = self.log_dir
        model_dir = os.path.join(log_dir, 'models')
        #print(model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        return model_dir

    @abstractproperty
    def lib_dir(self):
        pass

    # training setting
    @abstractproperty
    def num_epochs(self):
        pass

    @property
    def val_interval(self):
        '''
        Specify how many epochs between two validation steps
        Return <= 0 means no validation phase
        '''
        return 0

    @abstractmethod
    def create_optimizer(self, params) -> torch.optim.Optimizer:
        '''
        params (iterable): iterable of parameters to optimize or dicts defining
                           parameter groups
        '''
        pass

    @abstractmethod
    def create_lr_scheduler(self, optimizer:torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        pass

    @abstractmethod
    def create_loss_function(self) -> torch.nn.modules.loss._Loss:
        pass


    def create_attack_method(self, *inputs):
        '''
        Perform adversarial training against xxx adversary
        Return None means natural training
        '''
        return None

    # Evaluation Setting

    def create_evaluation_attack_method(self, *inputs):
        '''
        evaluating the robustness of model against xxx adversary
        Return None means only measuring clean accuracy
        '''
        return None




class SGDOptimizerMaker(object):

    def __init__(self, lr = 0.1, momentum = 0.9, weight_decay = 1e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, params):
        return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)


class PieceWiseConstantLrSchedulerMaker(object):

    def __init__(self, milestones:List[int], gamma:float = 0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

class IPGDAttackMethodMaker(object):

    def __init__(self, eps, sigma, nb_iters, norm, mean, std):
        self.eps = eps
        self.sigma = sigma
        self.nb_iters = nb_iters
        self.norm = norm
        self.mean = mean
        self.std = std

    def __call__(self, DEVICE):
        father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
        # print(father_dir)
        if not father_dir in sys.path:
            sys.path.append(father_dir)
        from attack.pgd import IPGD
        return IPGD(self.eps, self.sigma, self.nb_iters, self.norm, DEVICE, self.mean, self.std)

class LambdaLrSchedulerMaker(object):


    def __init__(self, func, last_epoch = -1):
        assert callable(func)

        self.func = func
        self.last_epoch = last_epoch

    def __call__(self, parameters):
        from torch.optim.lr_scheduler import LambdaLR
        lr_schduler = LambdaLR(parameters, self.func, self.last_epoch)
        return lr_schduler
