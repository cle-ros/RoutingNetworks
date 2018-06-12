"""
This file defines class MLP.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, nonlin=F.relu):
        nn.Module.__init__(self)
        self._layers = nn.ModuleList()
        input_dim = int(np.prod(input_dim))
        last_dim = input_dim
        self._nonlin = nonlin
        for hidden_layer_dim in layers:
            self._layers.append(nn.Linear(last_dim, hidden_layer_dim))
            last_dim = hidden_layer_dim
        self._layers.append(nn.Linear(last_dim, output_dim))

    def forward(self, arg, *args):
        out = arg
        for i in range(len(self._layers) - 1):
            layer = self._layers[i]
            evaluated = layer(out)
            out = self._nonlin(evaluated)
        out = self._layers[-1](out)
        return out

    __call__ = forward
