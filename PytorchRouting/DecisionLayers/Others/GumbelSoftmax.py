"""
This file defines class GumbelSoftmax.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/12/18
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..DecisionLayer import DecisionLayer


class GumbelSoftmax(DecisionLayer):
    """
    Class REINFORCE defines ...
    """
    def __init__(self, *args, **kwargs):
        DecisionLayer.__init__(self, *args, **kwargs)
        self._gumbel_softmax = GumbelSoftmax_()

    @staticmethod
    def _loss(sample):
        return torch.zeros(1).to(sample.state.device)

    def _forward(self, xs, mxs, agent):
        logits = self._policy[agent](xs)
        actions, multiples = self._gumbel_softmax.sample(logits)
        return xs*multiples, actions, logits


class GumbelSoftmax_(nn.Module):
    """
    Class GumbelSoftmax defines ...
    """

    def __init__(self, temperature_init=30, temperature_decay=0.9, hard=True, hook=None):
        nn.Module.__init__(self)
        self._temperature = temperature_init
        self._temperature_decay = temperature_decay
        self.softmax = nn.Softmax(dim=1)
        self._hard = hard
        self._hook = hook

    def reduce_temperature(self):
        self._temperature *= self._temperature_decay
        # print('The new temperature param is: {}'.format(self._temperature))

    def _sample_gumble(self, shape, eps=1e-20):
        U = torch.FloatTensor(*shape)
        U.uniform_(0, 1)
        logs = -torch.log(-torch.log(U + eps) + eps)
        return logs

    def _gumbel_softmax_sample(self, logits):
        y = logits + Variable(self._sample_gumble(logits.size())).to(logits.device)
        dist = self.softmax(y / self._temperature)
        if self._hook is not None:
            dist.register_hook(self._hook)
        return dist

    def forward(self, logits):
        y = self._gumbel_softmax_sample(logits)
        if self._hard:
            _, y_hard_index = torch.max(y, len(y.size())-1)
            y_hard = y.clone().data.zero_()
            y_hard[0, y_hard_index.squeeze()] = 1.
            y_no_grad = Variable(y.data)
            y = Variable(y_hard) - y_no_grad + y
        return y

    def sample(self, logits):
        y = self._gumbel_softmax_sample(logits)
        _, y_hard_index = torch.max(y, len(y.size()) -1)
        index = y_hard_index.data
        y_no_grad = Variable(y.data)
        multiplier = 1. + (y - y_no_grad).squeeze()[index]
        return index, multiplier

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
