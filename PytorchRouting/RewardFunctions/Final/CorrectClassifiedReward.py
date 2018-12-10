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
        # input checking - onehot vs indices
        if yest.numel() == yest.size(0):
            y_ind = yest
        else:
            _, y_ind = yest.max(dim=1)
        if ytrue.numel() == ytrue.size(0):
            yt_ind = ytrue
        else:
            _, yt_ind = ytrue.max(dim=1)
        return -1. + 2. * (y_ind.squeeze() == yt_ind.squeeze()).float()