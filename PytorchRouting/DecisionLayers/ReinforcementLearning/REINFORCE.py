"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch

from ..Decision import Decision


class REINFORCE(Decision):
    """
    REINFORCE (likelihood ratio policy gradient) based decision making.
    """
    @staticmethod
    def _loss(sample):
        return - sample.state.log_prob(sample.action) * sample.cum_return

    def _forward(self, xs, mxs, agent):
        policy = self._policy[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy)
        actions = distribution.sample()
        return xs, actions, distribution.logits
