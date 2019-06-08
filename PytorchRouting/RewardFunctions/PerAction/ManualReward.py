"""
This file defines class ManualReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 12/10/18
"""
import torch
from .PerActionBaseReward import PerActionBaseReward


class ManualReward(PerActionBaseReward):
    """
    Class ManualReward defines ...
    """

    def __init__(self, rewards, num_actions=None):
        PerActionBaseReward.__init__(self)
        if num_actions is not None:
            assert len(rewards) == num_actions
        self._rewards = torch.FloatTensor(rewards).squeeze()
        self._num_actions = num_actions

    def get_reward(self, dist, action):
        return self._rewards[action].to(action.device)
