"""
This file defines class CorrectClassifiedReward.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
from .BaseReward import BaseReward


class CorrectClassifiedReward(BaseReward):
    """
    Class CorrectClassifiedReward defines the +1 reward for correct classification, and -1 otherwise.
    """

    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)

    def forward(self, loss, yest, ytrue):
        _, y_max = yest.max(dim=1)
        _, yt_max = ytrue.max(dim=1)
        return -1. + 2. * (y_max.squeeze() == yt_max.squeeze()).float()