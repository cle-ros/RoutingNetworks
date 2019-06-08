"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch
import torch.nn.functional as F

from ..Decision import Decision


class AdvantageLearning(Decision):
    """
    QLearning (state-action value function) based decision making.
    """
    def __init__(self, *args, value_net=None, **kwargs):
        Decision.__init__(self, *args, **kwargs)
        self._value_mem = self._construct_policy_storage(
            1, self._pol_type, value_net, self._pol_hidden_dims)

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        value_loss = F.mse_loss(state[:, 0, 1], cum_return, reduction='none').view(-1)
        qval_loss = F.mse_loss(state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1),
                               cum_return - state[:, 0, 1].detach(), reduction='none')
        return value_loss + qval_loss

    def _forward(self, xs, agent):
        batch_dim = xs.size()[0]
        qvals = self._policy[agent](xs)
        value = self._value_mem[agent](xs)
        value = value.expand_as(qvals)
        exploration_dist = torch.ones(batch_dim, 2).float()
        exploration_dist[:, 0] *= 1-self._exploration
        exploration_dist[:, 1] *= self._exploration
        explore_bin = torch.multinomial(exploration_dist, 1).byte().to(xs.device)
        _, greedy = qvals.max(dim=1)
        if self.training:
            explore = torch.randint(low=0, high=qvals.size()[1], size=(batch_dim, 1)).to(xs.device).long()
            actions = torch.where(explore_bin, explore, greedy.unsqueeze(-1))
        else:
            actions = greedy
        state = torch.stack([qvals, value], 2)
        return xs, actions, explore_bin, state


class BootstrapAdvantageLearning(AdvantageLearning):
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        vtarget = torch.where(is_terminal, final_reward, next_state[:, 0, 1] + reward).detach()
        value_loss = F.mse_loss(state[:, 0, 1], vtarget, reduction='none').view(-1)
        atarget = (vtarget - state[:, 0, 1]).detach()
        adv_loss = F.mse_loss(state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1),
                               atarget, reduction='none')
        return value_loss + adv_loss


class SurpriseLearning(AdvantageLearning):
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        target = torch.where(is_terminal, final_reward, next_state[:, 0, 1] - reward).detach()
        value_loss = F.mse_loss(state[:, 0, 1], target, reduction='none').view(-1)
        qval_loss = F.mse_loss(state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1),
                               target - state[:, 0, 1].detach(), reduction='none')
        return value_loss + qval_loss
