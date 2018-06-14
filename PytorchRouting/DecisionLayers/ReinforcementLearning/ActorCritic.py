"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import copy
import torch
import torch.nn.functional as F

from ..Decision import Decision


class ActorCritic(Decision):
    """
    Class REINFORCE defines ...
    """

    def _construct_policy_storage(self, *args, **kwargs):
        Decision._construct_policy_storage(self, *args, **kwargs)
        self._value_mem = copy.deepcopy(self._policy)

    @staticmethod
    def _loss(sample):
        act_loss = - sample.state[:, 0, sample.action] * (sample.state[:, 1, sample.action] - sample.cum_return)
        ret_loss = F.smooth_l1_loss(sample.state[:, 1, sample.action], sample.cum_return).unsqueeze(-1)
        return act_loss + ret_loss

    def _forward(self, xs, mxs, agent):
        policy = self._policy[agent](xs)
        values = self._value_mem[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy)
        actions = distribution.sample()
        state = torch.stack([distribution.logits, values], 1)
        return xs, actions, state
