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
    Class REINFORCE defines ...
    """
    @staticmethod
    def _loss(sample):
        if sample.next_state is not None:
            target = torch.max(sample.next_state.data) - sample.reward
        else:
            target = sample.cum_return
        return F.smooth_l1_loss(sample.state[0, sample.action], target).unsqueeze(0)

    def _forward(self, xs, mxs, agent):
        batch_dim = xs.size()[0]
        policy = self._policy[agent](xs)
        exploration_dist = torch.ones(batch_dim, 2).float()
        exploration_dist[:, 0] *= 1-self._exploration
        exploration_dist[:, 1] *= self._exploration
        explore_bin = torch.multinomial(exploration_dist, 1).cuda()
        _, greedy = policy.max(dim=1)
        explore = torch.randint(low=0, high=policy.size()[1], size=(1, batch_dim)).cuda().long()
        actions = torch.where(explore_bin.byte(), explore, greedy.unsqueeze(-1))
        return xs, actions, [policy] * len(actions)
