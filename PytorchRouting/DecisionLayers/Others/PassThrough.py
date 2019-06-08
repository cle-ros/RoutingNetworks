"""
This file defines the pass through decision maker.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 2/7/19
"""
import torch

from ..Decision import Decision


class PassThrough(Decision):
    """
    This helper decision module does not actually make any decision, but is only a dummy useful for some
    implementations.
    """
    def __init__(self, *args, **kwargs):
        Decision.__init__(self, None, None, )

    @staticmethod
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        return torch.zeros(1).to(action.device)

    def _construct_policy_storage(self, _1, _2, _3, _4):
        return []

    def _forward(self, xs, prior_action): pass

    def forward(self, xs, mxs, _=None, __=None):
        return xs, mxs, torch.zeros(len(mxs), dtype=torch.long, device=xs.device)
