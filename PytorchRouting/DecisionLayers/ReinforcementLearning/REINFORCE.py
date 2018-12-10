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
        return - sample.state[:, sample.action] * sample.cum_return.detach()

    def _forward(self, xs, mxs, agent):
        policy = self._policy[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        return xs, actions, distribution.logits
