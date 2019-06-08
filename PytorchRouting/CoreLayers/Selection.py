"""
This file defines class Model.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/6/18
"""
import torch
import torch.nn as nn
# from torch.multiprocessing import Pool


class Selection(nn.Module):
    """
    Class RoutingWrapperModule defines a wrapper around a regular pytorch module that computes the actual routing
    given a list of modules to choose from, and a list of actions to select a module for each sample in a batch.
    """

    def __init__(self, *modules, name='', store_module_pointers=False):
        nn.Module.__init__(self)
        self.name = name
        # self._threads = threads
        self._submodules = nn.ModuleList(modules)
        self._selection_log = []
        self._logging_selections = False
        self.__output_dim = None
        self._store_module_pointers = store_module_pointers

    def forward(self, xs, mxs, actions, mask=None):
        """
        This method takes a list of samples - a batch - and calls _forward_sample on each. Samples are
        a tensor where the first dimension is the batch dimension.
        :param xs:
        :param mxs:
        :param actions:
        :param mask: a torch.ByteTensor that determines if the trajectory is active. if it is not, no action
                               will be executed
        :return:
        """
        assert len(xs) == len(mxs)
        batch_size = xs.size(0)
        # capture the special case of just one submodule - and skip all computation
        if len(self._submodules) == 1:
            return self._submodules[0](xs), mxs, actions
        # retrieving output dim for output instantiation
        if self.__output_dim is None:
            self.__output_dim = self._submodules[0](xs[0].unsqueeze(0)).shape[1:]
        # initializing the "termination" mask
        mask = torch.ones(batch_size, dtype=torch.uint8, device=xs.device) \
            if mask is None else mask
        # parallelizing this loop does not work. however, we can split the batch by the actions
        # creating the target variable
        ys = torch.zeros((batch_size, *self.__output_dim), dtype=torch.float, device=xs.device)
        for i in torch.arange(actions.max() + 1, device=xs.device):
            if i not in actions:
                continue
            # computing the mask as the currently active action on the active trajectories
            m = ((actions == i) * mask)
            if not any(m):
                continue
            ys[m] = self._submodules[i](xs[m])
        if self._logging_selections:
            self._selection_log += actions.reshape(-1).cpu().tolist()
        if self._store_module_pointers:
            for mx, a in zip(mxs, actions):
                mx.append('selected_modules', self._submodules[a])
        return ys, mxs, actions

    def start_logging_selections(self):
        self._logging_selections = True

    def stop_logging_and_get_selections(self, add_to_old=False):
        self._logging_selections = False
        logs = list(set([int(s) for s in self._selection_log]))
        del self._selection_log[:]
        self.last_selection_freeze = logs + self.last_selection_freeze if add_to_old else logs
        return self.last_selection_freeze
