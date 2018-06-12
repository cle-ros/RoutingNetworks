"""
This file defines class NegLossReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
from .BaseReward import BaseReward


class NegLossReward(BaseReward):
    """
    Class NegLossReward defines ...
    """

    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)

    def forward(self, loss):
        return -loss.data