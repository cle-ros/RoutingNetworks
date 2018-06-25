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
    ActorCritic based decision making.
    """

    def _construct_policy_storage(self, *args, **kwargs):
        Decision._construct_policy_storage(self, *args, **kwargs)
        self._value_mem = copy.deepcopy(self._policy)

    @staticmethod
    def _loss(sample):
        act_loss = - sample.state[:, sample.action, 0] * (sample.state[:, sample.action, 1] - sample.cum_return)
        ret_loss = F.smooth_l1_loss(sample.state[:, sample.action, 1], sample.cum_return).unsqueeze(-1)
        return act_loss + ret_loss

    def _forward(self, xs, mxs, agent):
        policy = self._policy[agent](xs)
        values = self._value_mem[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        state = torch.stack([distribution.logits, values], 2)
        return xs, actions, state
