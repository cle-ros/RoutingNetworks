"""
This file defines class DecisionModule.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import abc
import copy
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from .PolicyStorage import ApproxPolicyStorage, TabularPolicyStorage
from PytorchRouting.RewardFunctions.PerAction.PerActionBaseReward import PerActionBaseReward


class DecisionLayer(nn.Module, metaclass=abc.ABCMeta):
    """
    Class DecisionModule defines ...
    """

    def __init__(
            self,
            num_selections,
            in_features,
            num_agents=1,
            exploration=0.1,
            policy_storage_type='approx',
            detach=True,
            approx_hidden_dims=(),
            approx_module=None,
            additional_reward_class=PerActionBaseReward,
            additional_reward_args={}
        ):
        nn.Module.__init__(self)
        self._in_features = in_features
        self._num_selections = num_selections
        self._num_agents = num_agents
        self._exploration = exploration
        self._detach = detach
        self._construct_policy_storage(policy_storage_type, approx_module, approx_hidden_dims)
        self.additional_reward_func = additional_reward_class(*additional_reward_args)

    @abc.abstractmethod
    def _forward(self, xs, mxs, prior_action): return torch.FloatTensor(1, 1), [], torch.FloatTensor(1, 1)

    @staticmethod
    @abc.abstractmethod
    def _loss(sample):
        pass

    def _construct_policy_storage(self, policy_storage_type, approx_module, approx_hidden_dims):
        if policy_storage_type in ('approx', 0):
            if approx_module:
                self._policy = nn.ModuleList(
                    [ApproxPolicyStorage(approx=copy.deepcopy(approx_module), detach=self._detach)
                     for _ in range(self._num_agents)]
                )
            else:
                self._policy = nn.ModuleList(
                    [ApproxPolicyStorage(
                        in_features=self._in_features,
                        num_selections=self._num_selections,
                        hidden_dims=approx_hidden_dims,
                        detach=self._detach)
                        for _ in range(self._num_agents)]
                )
        else:
            self._policy = nn.ModuleList(
                [TabularPolicyStorage(num_selections=self._num_selections)
                for _ in range(self._num_agents)]
            )

    def forward(self, xs, mxs, prior_actions=None):
        """
        The forward method of DecisionModule takes a batch of inputs, and a list of metainformation, and
        append the decision made to the metainformation objects.
        :param xs:
        :param mxs:
        :return:
        """
        if self._num_agents > 1:
            if prior_actions is None:
                raise ValueError('If multiple agents are available, argument '
                                 '`prior_actions` must be provided as a long Tensor of size '
                                 '(batch_size),\nwhere each entry determines the agent for '
                                 'that sample.')
            actions, dists, ys = [], [], []
            for x, mx, pa in zip(xs.split(1, dim=0), mxs, prior_actions):
                y, action, generating_dist = self._forward(x, [mx], pa)
                actions.append(action)
                dists.append(generating_dist)
                ys.append(y)
            actions = torch.cat(actions, dim=0)
            dists = torch.cat(dists, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            ys, actions, dists = self._forward(xs, mxs, 0)
        for a, d, mx in zip(actions, dists.split(1, dim=0), mxs):
            mx.append('actions', a)
            mx.append('states', d)
            mx.append('loss_funcs', self._loss)
            mx.append('reward_func', self.additional_reward_func)
            # if len(d.size()) > 2 and d.size()[1] == 2:
            #     self.additional_reward_func.register(d[:, 0], actions)
            # else:
            self.additional_reward_func.register(d, a)
        return ys, mxs, actions

