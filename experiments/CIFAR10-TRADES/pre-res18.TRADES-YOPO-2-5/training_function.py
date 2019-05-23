import torch
import torch.nn as nn
from config import config

from loss import Hamiltonian, cal_l2_norm
import torch.nn.functional as F

from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm


class FastGradientLayerOneTrainer(object):

    def __init__(self, Hamiltonian_func, param_optimizer,
                    inner_steps=2, sigma = 0.008, eps = 0.03):
        self.inner_steps = inner_steps
        self.sigma = sigma
        self.eps = eps
        self.Hamiltonian_func = Hamiltonian_func
        self.param_optimizer = param_optimizer

    def step(self, inp, p, eta):
        '''
        Perform Iterative Sign Gradient on eta
        ret: inp + eta
        '''

        p = p.detach()

        for i in range(self.inner_steps):
            tmp_inp = inp + eta
            tmp_inp = torch.clamp(tmp_inp, 0, 1)
            H = self.Hamiltonian_func(tmp_inp, p)

            eta_grad = torch.autograd.grad(H, eta, only_inputs=True, retain_graph=False)[0]
            eta_grad_sign = eta_grad.sign()
            eta = eta - eta_grad_sign * self.sigma

            eta = torch.clamp(eta, -1.0 * self.eps, self.eps)
            eta = torch.clamp(inp + eta, 0.0, 1.0) - inp
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()


        yofo_inp = eta + inp
        yofo_inp = torch.clamp(yofo_inp, 0, 1)
        loss = -1.0 * (self.Hamiltonian_func(yofo_inp, p) -
                       config.weight_decay * cal_l2_norm(self.Hamiltonian_func.layer))

        loss.backward()


        return yofo_inp, eta


def train_one_epoch(net, batch_generator, optimizer,
                    criterion, LayerOneTrainner, K,
                    DEVICE=torch.device('cuda:0'),descrip_str='Training'):

    net.train()
    pbar = tqdm(batch_generator)
    yofoacc = -1
    pbar.set_description(descrip_str)

    trades_criterion = torch.nn.KLDivLoss(size_average=False) #.to(DEVICE)

    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        net.eval()
        eta = 0.001 * torch.randn(data.shape).cuda().detach().to(DEVICE)

        eta.requires_grad_()


        raw_soft_label = F.softmax(net(data), dim=1).detach()
        for j in range(K):
            pred = net(data + eta.detach())

            with torch.enable_grad():
                loss = trades_criterion(F.log_softmax(pred, dim = 1), raw_soft_label)#raw_soft_label.detach())

            p = -1.0 * torch.autograd.grad(loss, [net.layer_one_out, ])[0]

            yofo_inp, eta = LayerOneTrainner.step(data, p, eta)

            with torch.no_grad():

                if j == K - 1:
                    yofo_pred = net(yofo_inp)
                    yofo_loss = criterion(yofo_pred, label)
                    yofoacc = torch_accuracy(yofo_pred, label, (1,))[0].item()


        net.train()

        optimizer.zero_grad()
        LayerOneTrainner.param_optimizer.zero_grad()

        raw_pred = net(data)
        acc = torch_accuracy(raw_pred, label, (1,))
        clean_acc = acc[0].item()
        clean_loss = criterion(raw_pred, label)


        adv_pred = net(torch.clamp(data + eta.detach(), 0.0, 1.0))
        kl_loss = trades_criterion(F.log_softmax(adv_pred, dim=1),
                                    F.softmax(raw_pred, dim=1)) / data.shape[0]

        loss = clean_loss + kl_loss
        loss.backward()

        optimizer.step()
        LayerOneTrainner.param_optimizer.step()

        optimizer.zero_grad()
        LayerOneTrainner.param_optimizer.zero_grad()

        pbar_dic = OrderedDict()
        pbar_dic['Acc'] = '{:.2f}'.format(clean_acc)
        pbar_dic['cleanloss'] = '{:.3f}'.format(clean_loss.item())
        pbar_dic['klloss'] = '{:.3f}'.format(kl_loss.item())
        pbar_dic['YofoAcc'] = '{:.2f}'.format(yofoacc)
        pbar_dic['Yofoloss'] = '{:.3f}'.format(yofo_loss.item())
        pbar.set_postfix(pbar_dic)

    return clean_acc, yofoacc