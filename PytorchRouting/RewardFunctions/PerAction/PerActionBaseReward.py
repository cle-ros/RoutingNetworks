"""
This file defines class BaseReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import abc
import torch


class PerActionBaseReward(object, metaclass=abc.ABCMeta):
    """
    Class BaseReward defines the base class for per-action rewards.
    """

    def __init__(self):
        self._dists = []
        self._actions = []
        self._precomp = None

    def register(self, dist, action):
        self._dists.append(dist)
        self._actions.append(action.data)

    def clear(self):
        del self._dists[:]
        del self._actions[:]
        self._precomp = None

    def get_reward(self, dist, action):
        return torch.FloatTensor([0.]).to(action.device)
