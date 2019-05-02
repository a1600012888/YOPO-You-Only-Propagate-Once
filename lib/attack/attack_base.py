import torch
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

class AttackBase(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, net, inp, label, target = None):
        '''

        :param inp: batched images
        :param target: specify the indexes of target class, None represents untargeted attack
        :return: batched adversaril images
        '''
        pass

    @abstractmethod
    def to(self, device):
        pass



def clip_eta(eta, norm, eps, DEVICE = torch.device('cuda:0')):
    '''
    helper functions to project eta into epsilon norm ball
    :param eta: Perturbation tensor (should be of size(N, C, H, W))
    :param norm: which norm. should be in [1, 2, np.inf]
    :param eps: epsilon, bound of the perturbation
    :return: Projected perturbation
    '''

    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    with torch.no_grad():
        avoid_zero_div = torch.tensor(1e-12).to(DEVICE)
        eps = torch.tensor(eps).to(DEVICE)
        one = torch.tensor(1.0).to(DEVICE)

        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            normalize = torch.norm(eta.reshape(eta.size(0), -1), p = norm, dim = -1, keepdim = False)
            normalize = torch.max(normalize, avoid_zero_div)

            normalize.unsqueeze_(dim = -1)
            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)

            factor = torch.min(one, eps / normalize)
            eta = eta * factor
    return eta

def test_clip():

    a = torch.rand((10, 3, 28, 28)).cuda()

    epss = [0.1, 0.5, 1]

    norms = [1, 2, np.inf]
    for e, n in zip(epss, norms):
        print(e, n)
        c = clip_eta(a, n, e, True)

        print(c)

if __name__ == '__main__':
    test_clip()
