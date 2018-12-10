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


class RunningAverageCollaborationReward(PerActionBaseReward):
    """
    Provides the same functionality as CollaborationReward, but with a much faster computing running average.
    """

    def __init__(self, reward_ratio=0.1, num_actions=None, history_len=256):
        PerActionBaseReward.__init__(self, history_len)
        self._reward_ratio = reward_ratio
        self._num_actions = num_actions
        self._adaptation_rate = 10 ** (-1./history_len)
        self._dists = None
        self._actions = None
        self._precomp = None

    def register(self, dist, action):
        # initializing
        if self._actions is None:
            self._actions = torch.zeros(self._num_actions).to(dist.device)
        # one hot encoding
        action_oh = torch.zeros_like(self._actions).float()
        action_oh[action.item()] = 1.
        # running average learning
        self._actions = self._adaptation_rate * self._actions + (1 - self._adaptation_rate) * action_oh
        # normalizing
        self._actions = self._actions / self._actions.sum()

    def clear(self):
        self._dists = None
        self._actions = None
        self._precomp = None

    def get_reward(self, dist, action):
        return self._actions[action]
