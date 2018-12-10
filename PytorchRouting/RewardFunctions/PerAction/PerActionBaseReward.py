"""
This file defines class BaseReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
from collections import deque
import abc
import torch


class PerActionBaseReward(object, metaclass=abc.ABCMeta):
    """
    Class BaseReward defines the base class for per-action rewards.
    """

    def __init__(self, history_window=256, *args, **kwargs):
        self._hist_len = history_window
        self._dists = deque(maxlen=history_window)
        self._actions = deque(maxlen=history_window)
        self._precomp = None

    def register(self, dist, action):
        self._dists.append(dist.detach())
        self._actions.append(action.detach())

    def clear(self):
        self._dists = deque(maxlen=self._hist_len)
        self._actions = deque(maxlen=self._hist_len)
        self._precomp = None

    def get_reward(self, dist, action):
        return torch.FloatTensor([0.]).to(action.device)
