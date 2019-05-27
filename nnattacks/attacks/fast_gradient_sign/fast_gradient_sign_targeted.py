# -*- coding: utf-8 -*-
import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients

from nnattacks.base import BaseAttack


class FastGradientSignTargeted(BaseAttack):
    def __init__(self,
                 alpha: float = 0.025,
                 n_iter: int = 10,
                 eps: float = 0.25,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 device=torch.device("cpu")):

        self.alpha = alpha
        self.eps = eps
        self.n_iter = n_iter
        self.criterion = criterion
        self.device = device

    def generate_perturbation(self,
                              model: nn.Module,
                              x: torch.Tensor,
                              y_target: torch.Tensor):

        x_adv = x.clone().to(self.device)
        x_adv.requires_grad = True

        with tqdm.tqdm(range(self.n_iter), ncols=100) as pbar:
            for i in pbar:
                zero_gradients(x_adv)
                output = model(x_adv)

                loss = self.criterion(output, y_target)
                loss.backward()

                x_grad = self.alpha * torch.sign(x_adv.grad.data)
                adv = x_adv.data - x_grad

                grad = adv - x_adv
                grad = torch.clamp(grad, -self.eps, self.eps)
                x_adv.data += grad

                pbar.set_postfix(OrderedDict(
                    loss="{:.8f}".format(loss.item())
                ))

        return x_adv, grad
