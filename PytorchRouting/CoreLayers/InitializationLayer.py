"""
This file defines class RoutingTechnicalLayers.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import torch.nn as nn

from PytorchRouting.Helpers.Meta import Meta


class InitializationLayer(nn.Module):
    """
    Class RoutingTechnicalLayers defines ...
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, xs, tasks=()):
        if tasks:
            mxs = [Meta(task=t) for t in tasks]
        else:
            mxs = [Meta() for _ in xs]
        return xs, mxs, None
