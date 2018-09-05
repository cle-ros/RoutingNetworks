"""
This file defines class CollaborationReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import torch

from .PerActionBaseReward import PerActionBaseReward


class CollaborationReward(PerActionBaseReward):
    """
    Class CollaborationReward defines a collaboration reward measured by the average probability
    of taking the action taken by an agent.
    """

    def __init__(self, reward_ratio=0.1, num_actions=None, history_len=256):
        PerActionBaseReward.__init__(self, history_len)
        self._reward_ratio = reward_ratio
        self._num_actions = num_actions

    def get_reward(self, dist, action):
        action_count = torch.zeros(len(self._actions), self._num_actions).to(dist.device)
        action_count = action_count.scatter(1, torch.stack(list(self._actions), 0).unsqueeze(1), 1.)
        action_count = torch.sum(action_count, dim=0)/len(self._actions)
        self._precomp = action_count
        self._precomp = self._reward_ratio * self._precomp
        return self._precomp[action]

