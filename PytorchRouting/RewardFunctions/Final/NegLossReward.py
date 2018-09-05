"""
This file defines class NegLossReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import torch
from .BaseReward import BaseReward


class NegLossReward(BaseReward):
    """
    Class NegLossReward defines the simplest reward function, expressed as the negative loss.
    """

    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)

    def forward(self, loss, yest, ytrue):
        with torch.no_grad():
            reward = -loss.squeeze()
        return reward