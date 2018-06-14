"""
This file defines class RoutingTechnicalLayers.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import torch.nn as nn

from PytorchRouting.Helpers.SampleMetaInformation import SampleMetaInformation


class Initialization(nn.Module):
    """
    The initialization class defines a thin layer that initializes the meta-information and actions - composing
    the pytorch-routing information triplet.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, xs, tasks=()):
        if tasks:
            mxs = [SampleMetaInformation(task=t) for t in tasks]
        else:
            mxs = [SampleMetaInformation() for _ in xs]
        return xs, mxs, None
