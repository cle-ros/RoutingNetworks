"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch.nn.functional as F

from .QLearning import QLearning


class SARSA(QLearning):
    """
    SARSA on-policy q-function learning.
    """
    @staticmethod
    def _loss(sample):
        if sample.next_action is not None:
            target = sample.next_state.data[sample.next_action] - sample.reward
        else:
            target = sample.cum_return
        return F.smooth_l1_loss(sample.state[0, sample.action], target).unsqueeze(0)
