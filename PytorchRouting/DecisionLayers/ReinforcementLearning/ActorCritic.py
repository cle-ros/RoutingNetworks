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
    def __init__(self, *args, **kwargs):
        Decision.__init__(self, *args, **kwargs)
        self._value_mem = self._construct_policy_storage(
            self._num_selections, self._pol_type, None, self._pol_hidden_dims)

    @staticmethod
    def _loss(sample):
        normalized_return = (sample.cum_return - sample.state[:, sample.action, 1]).detach()
        act_loss = - sample.state[:, sample.action, 0] * normalized_return
        if sample.next_state is not None:
            value_target = torch.max(sample.next_state) - sample.reward
        else:
            value_target = sample.cum_return
        value_target = value_target.detach()
        val_loss = F.mse_loss(sample.state[:, sample.action, 1], value_target).unsqueeze(-1)
        return act_loss + val_loss

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
