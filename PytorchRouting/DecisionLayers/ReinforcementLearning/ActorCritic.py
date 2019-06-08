"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import copy
import torch
import torch.nn.functional as F

from .REINFORCE import REINFORCE


class ActorCritic(REINFORCE):
    """
    ActorCritic based decision making.
    """
    def __init__(self, *args, qvalue_net=None, **kwargs):
        REINFORCE.__init__(self, *args, **kwargs)
        if qvalue_net is None and 'policy_net' in kwargs:
            qvalue_net = copy.deepcopy(kwargs['policy_net'])
        self._qvalue_mem = self._construct_policy_storage(
            self._num_selections, self._pol_type, qvalue_net, self._pol_hidden_dims)

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        normalized_return = (cum_return - state[:, :, 1].gather(index=action.unsqueeze(1), dim=1).view(-1)).detach()
        act_loss = - state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1) * normalized_return
        value_target = torch.where(is_terminal, final_reward, next_state[:, :, 1].max(dim=1)[0] - reward).detach()
        val_loss = F.mse_loss(state[:, :, 1].gather(index=action.unsqueeze(1), dim=1).view(-1),
                              value_target, reduction='none').view(-1)
        return act_loss + val_loss

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        values = self._qvalue_mem[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy/self._pg_temperature)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        state = torch.stack([distribution.logits, values], 2)
        return xs, actions, self._eval_stochastic_are_exp(actions, state), state
