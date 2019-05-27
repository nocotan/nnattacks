# -*- coding: utf-8 -*-
import abc


class BaseAttack(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_perturbation(self, network):
        raise NotImplementedError
