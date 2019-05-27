# -*- coding: utf-8 -*-
import tqdm
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients

from nnattacks.base import BaseAttack


class FastGradientSignUntargeted(BaseAttack):
    def __init__(self,
                 eps: float = 0.04,
                 n_iter: int = 10,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 device=torch.device("cpu")):

        self.eps = eps
        self.n_iter = n_iter
        self.criterion = criterion
        self.device = device

    def generate_perturbation(self,
                              model: nn.Module,
                              x: torch.Tensor,
                              y: torch.Tensor):

        x_adv = x.clone().to(self.device)
        x_adv.requires_grad = True

        with tqdm.tqdm(range(self.n_iter), ncols=100) as pbar:
            for i in pbar:
                zero_gradients(x_adv)
                output = model(x_adv)

                loss = self.criterion(output, y)
                loss.backward(retain_graph=True)

                grad = torch.sign(x_adv.grad.data)
                x_adv.data += (self.eps * grad)

        return x_adv, grad
