"""
This file defines class GumbelSoftmax.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/12/18
"""
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..Decision import Decision


class GumbelSoftmax(Decision):
    """
    Class GumbelSoftmax defines a decision making procedure that uses the GumbelSoftmax reparameterization trick
    to perform differentiable sampling from the categorical distribution.
    """
    def __init__(self, *args, **kwargs):
        Decision.__init__(self, *args, **kwargs)
        # translating exploration into the sampling temperature parameter in [0.1, 10]
        self._gumbel_softmax = GumbelSoftmaxSampling(temperature_init=1)
        self.set_exploration(self._exploration)

    def set_exploration(self, exploration):
        temperature = 0.1 + 9.9*exploration
        self._gumbel_softmax.set_temperature(temperature)

    @staticmethod
    def _loss(sample):
        return torch.zeros(1).to(sample.state.device)

    def _forward(self, xs, mxs, agent):
        logits = self._policy[agent](xs)
        if self.training:
            actions, multiples = self._gumbel_softmax.sample(logits)
        else:
            actions = logits.max(dim=1)[1]
            multiples = 1.
        ys = (xs.view(xs.size(0), -1) * multiples).contiguous().view(xs.shape)  # shape casting to allow for mask-multiply
        return ys, actions, logits


class GumbelSoftmaxSampling(nn.Module):
    """
    This class defines the core functionality to sample from a gumbel softmax distribution
    """

    def __init__(self, temperature_init=1., hard=True, hook=None):
        nn.Module.__init__(self)
        self._temperature = temperature_init
        self.softmax = nn.Softmax(dim=1)
        self._hard = hard
        self._hook = hook

    def set_temperature(self, temperature):
        self._temperature = temperature
        # print('The new temperature param is: {}'.format(self._temperature))

    @staticmethod
    def _sample_gumble(shape, eps=1e-20):
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
            y_no_grad = y.detach()
            y = y_hard - y_no_grad + y
        return y

    def sample(self, logits):
        y = self._gumbel_softmax_sample(logits)
        _, y_hard_index = torch.max(y, dim=-1)
        index = y_hard_index.detach().view(-1, 1)
        y_fake_grad = y - y.detach()
        # if len(y_fake_grad.shape) == 1:
        #     y_fake_grad = y_fake_grad.view(1, -1)
        multiplier = 1 + torch.gather(y_fake_grad, 1, index)
        return index, multiplier

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
