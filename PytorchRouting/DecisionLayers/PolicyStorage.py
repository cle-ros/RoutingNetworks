"""
This file defines class ApproxPolicyStorageDecisionModule.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import abc
import torch
import torch.nn as nn
from torch.autograd import Variable
from PytorchRouting.Helpers.MLP import MLP


class ApproxPolicyStorage(nn.Module, metaclass=abc.ABCMeta):
    """
    Class ApproxPolicyStorage defines a simple module to store a policy approximator.
    """
    def __init__(self, approx=None, in_features=None, num_selections=None, hidden_dims=(), detach=True):
        nn.Module.__init__(self)
        self._detach = detach
        if approx:
            self._approx = approx
        else:
            self._approx = MLP(
                in_features,
                num_selections,
                hidden_dims
            )

    def forward(self, xs):
        if self._detach:
            xs = Variable(xs.data)
        policies = self._approx(xs)
        return policies


class TabularPolicyStorage(nn.Module, metaclass=abc.ABCMeta):
    """
    Class TabularPolicyStorage defines a simple module to store a policy in tabular form.
    """
    def __init__(self, approx=None, in_features=None, num_selections=None, hidden_dims=()):
        nn.Module.__init__(self)
        self._approx = nn.Parameter(
            torch.ones(1, num_selections).float()/num_selections
        )

    def forward(self, xs):
        policies = self._approx
        return policies
