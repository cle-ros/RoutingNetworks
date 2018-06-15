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

    def __init__(self, reward_ratio=0.1, num_actions=None):
        PerActionBaseReward.__init__(self)
        self._reward_ratio = reward_ratio
        self._num_actions = num_actions

    def get_reward(self, dist, action):
        if self._precomp is None:
            action_count = torch.zeros(len(self._actions), self._num_actions).cuda()
            action_count = action_count.scatter(1, torch.stack(self._actions, 0).unsqueeze(1), 1.)
            action_count = torch.sum(action_count, dim=0)/len(self._actions)
            self._precomp = action_count
        return self._precomp[action]

