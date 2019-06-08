"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch
import torch.nn.functional as F

from .QLearning import QLearning


class SARSA(QLearning):
    """
    SARSA on-policy q-function learning.
    """
    # target = torch.where(is_terminal, reward, next_state[:, :, 0].max(dim=1)[0] - reward)
    # target = target.detach()
    # return F.mse_loss(state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1),
    #                   target.view(-1), reduction='none')

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        target = torch.where(is_terminal, final_reward,
                             state[:, :, 0].gather(index=next_action.unsqueeze(1), dim=1).
                             view(-1)).detach()
        return F.mse_loss(state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1),
                          target.view(-1), reduction='none')
