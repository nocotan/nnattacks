# -*- coding: utf-8 -*-
import abc
import torch
import torch.nn as nn


class BaseAttack(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_perturbation(self, model: nn.Module, x: torch.Tensor):
        raise NotImplementedError
