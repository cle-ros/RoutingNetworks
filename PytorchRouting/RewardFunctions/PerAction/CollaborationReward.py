"""
This file defines class CollaborationReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import torch

from .PerActionBaseReward import PerActionBaseReward


class CollaborationReward(PerActionBaseReward):
    """
    Class CollaborationReward defines ...
    """

    def __init__(self, *args, **kwargs):
        PerActionBaseReward.__init__(self, *args, **kwargs)

    def get_reward(self, dist, action):
        if self._precomp is None:
            self._precomp = torch.mean(torch.cat(self._dists, dim=0), dim=0)
        return self._precomp[action]

