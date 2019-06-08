"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch
import torch.nn.functional as F

from ..Decision import Decision


class QLearning(Decision):
    """
    QLearning (state-action value function) based decision making.
    """

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        target = torch.where(is_terminal, final_reward, next_state[:, :, 0].max(dim=1)[0] - reward).detach()
        return F.mse_loss(state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1),
                          target.view(-1), reduction='none')

    def _forward(self, xs, agent):
        batch_dim = xs.size()[0]
        policy = self._policy[agent](xs)
        exploration_dist = torch.ones(batch_dim, 2).float()
        exploration_dist[:, 0] *= 1-self._exploration
        exploration_dist[:, 1] *= self._exploration
        explore_bin = torch.multinomial(exploration_dist, 1).byte().to(xs.device)
        _, greedy = policy.max(dim=1)
        if self.training:
            explore = torch.randint(low=0, high=policy.size()[1], size=(batch_dim, 1)).to(xs.device).long()
            actions = torch.where(explore_bin, explore, greedy.unsqueeze(-1))
        else:
            actions = greedy
        return xs, actions, explore_bin, policy
