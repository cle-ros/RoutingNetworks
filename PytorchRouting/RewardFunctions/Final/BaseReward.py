"""
This file defines class BaseReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaseReward(nn.Module, metaclass=abc.ABCMeta):
    """
    Class BaseReward defines ...
    """

    def __init__(self, scale=1.):
        nn.Module.__init__(self)
        self._scale = scale

    @abc.abstractmethod
    def forward(self, loss): pass
