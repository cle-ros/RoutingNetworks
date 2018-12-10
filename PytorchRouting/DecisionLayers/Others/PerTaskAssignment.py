"""
This file defines class PerTaskAssignment.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/12/18
"""
"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch

from ..Decision import Decision


class PerTaskAssignment(Decision):
    """
    This simple class translates task assignments stored in the meta-information objects to actions.
    """
    def __init__(self, *args, **kwargs):
        Decision.__init__(self, None, None, )

    @staticmethod
    def _loss(sample):
        return torch.zeros(1).to(sample.action.device)

    def _construct_policy_storage(self, _1, _2, _3, _4):
        return []

    def _forward(self, xs, mxs, agent):
        actions = torch.LongTensor([m.task for m in mxs]).to(xs.device)
        return xs, actions, torch.zeros(xs.size()[0], 1)
