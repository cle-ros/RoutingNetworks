"""
This file defines class RoutingSequential.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/13/18
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PytorchRouting.CoreLayers.InitializationLayer import InitializationLayer


class Sequential(nn.Sequential):
    """
    Sequential is a routing wrapper around the original torch.nn.Sequential class.
    It includes the "Initialization" layer and handles the routing triplet (y, meta, actions) sequentially.
    As a consequence, it cannot handle cases where actions are not immediately consumed, but have to be
    handled repeatedly (dispatching).
    """

    def __init__(self, *args):
        additional_modules = OrderedDict([('initialization', InitializationLayer())])
        if isinstance(args, OrderedDict):
            args = additional_modules.update(args)
        else:
            args = list(additional_modules.values()) + list(args)
        nn.Sequential.__init__(self, *args)

    def forward(self, x, tasks=()):
        """
        As the class Sequential includes an initialization layer, forward only takes a batch of input, and a list of
        tasks.
        :param x: samples. the first dim has to be the batch dimension
        :param tasks: a list/tuple/iterable of integer task labels
        :return:
        """
        initialization_module = self._modules[list(self._modules.keys())[0]]
        ys, meta, actions = initialization_module(x, tasks=tasks)
        for name, mod in list(self._modules.items())[1:]:
            ys, meta, actions = mod(ys, meta, actions)
        return ys, meta
