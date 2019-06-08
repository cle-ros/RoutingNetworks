"""
This file defines class DecisionModule.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import abc
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.distribution import Distribution

from .PolicyStorage import ApproxPolicyStorage, TabularPolicyStorage
from PytorchRouting.RewardFunctions.PerAction.PerActionBaseReward import PerActionBaseReward


class Decision(nn.Module, metaclass=abc.ABCMeta):
    """
    Class DecisionModule defines the base class for all decision modules.
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
            policy_net=None,
            additional_reward_func=PerActionBaseReward(),
            set_pg_temp=False,
            **kwargs
        ):
        nn.Module.__init__(self)
        self._in_features = in_features
        self._num_selections = num_selections
        self._num_agents = num_agents
        self._exploration = exploration
        self._detach = detach
        self._pol_type = policy_storage_type
        self._pol_hidden_dims = approx_hidden_dims
        self._policy = self._construct_policy_storage(
            self._num_selections, self._pol_type, policy_net, self._pol_hidden_dims)
        self.additional_reward_func = additional_reward_func
        self._dist_dim = 1
        self._set_pg_temp = set_pg_temp
        self._pg_temperature = 1.

    def set_exploration(self, exploration):
        self._exploration = exploration

    @abc.abstractmethod
    def _forward(self, xs, prior_action):
        return torch.zeros(1, 1), [], torch.zeros(1, 1)

    @staticmethod
    def _eval_stochastic_are_exp(actions, dist):
        if len(dist.shape) == 3:
            dist = dist[:, :, 0]
        return (torch.max(dist, dim=1)[1].view(-1) == actions.view(-1)).byte()

    @abc.abstractmethod
    def _forward(self, xs, prior_action):
        return torch.zeros(1, 1), [], torch.zeros(1, 1)

    @staticmethod
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        pass

    def _construct_policy_storage(self, out_dim, policy_storage_type, approx_module, approx_hidden_dims, in_dim=None):
        in_dim = in_dim or self._in_features
        if approx_module is not None:
            policy = nn.ModuleList(
                [ApproxPolicyStorage(approx=copy.deepcopy(approx_module), detach=self._detach)
                 for _ in range(self._num_agents)]
            )
        elif policy_storage_type in ('approx', 0):
            policy = nn.ModuleList(
                [ApproxPolicyStorage(
                    in_features=in_dim,
                    num_selections=out_dim,
                    hidden_dims=approx_hidden_dims,
                    detach=self._detach)
                    for _ in range(self._num_agents)]
            )
        elif policy_storage_type in ('tabular', 1):
            policy = nn.ModuleList(
                [TabularPolicyStorage(num_selections=out_dim)
                for _ in range(self._num_agents)]
            )
        else:
            raise ValueError(f'Policy storage type {policy_storage_type} not understood.')
        return policy

    def forward(self, xs, mxs, prior_actions=None, mask=None, update_target=None):
        """
        The forward method of DecisionModule takes a batch of inputs, and a list of metainformation, and
        append the decision made to the metainformation objects.
        :param xs:
        :param mxs:
        :param prior_actions: prior actions that select the agent
        :param mask: a torch.ByteTensor that determines if the trajectory is active. if it is not, no action
                               will be executed
        :param update_target: (only relevant for GumbelSoftmax) if specified, this will include the gradientflow
                              in update_target, and will thus return update_target
        :return: xs OR update_target, if specified, with potentially an attached backward object
        """
        # input checking
        assert len(xs) == len(mxs)
        batch_size = xs.size(0)
        assert self._num_agents == 1 or prior_actions is not None, \
            'Decision makers with more than one action have to have prior_actions provided.'
        assert mask is None or mask.max() == 1, \
            'Please check that a batch being passed in has at least one active (non terminated) trajectory.'
        # computing the termination mask and the prior actions if not passed in
        mask = torch.ones(batch_size, dtype=torch.uint8, device=xs.device) \
            if mask is None else mask
        prior_actions = torch.zeros(batch_size, dtype=torch.long, device=xs.device) \
            if prior_actions is None or len(prior_actions) == 0 else prior_actions.reshape(-1)
        ys = xs.clone() if update_target is None else update_target.clone()  # required as in-place ops follow
        # initializing the return vars
        actions = torch.zeros(batch_size, dtype=torch.long, device=xs.device)
        are_exp = torch.zeros(batch_size, dtype=torch.uint8, device=xs.device)
        dists = torch.zeros((batch_size, self._num_selections, 5), device=xs.device)
        # "clustering" by agent
        for i in torch.arange(0, prior_actions.max() + 1, device=xs.device):
            if i not in prior_actions:
                continue
            # computing the mask as the currently computed agent on the active trajectories
            m = ((prior_actions == i) * mask)
            if not any(m):
                continue
            # selecting the actions
            y, a, e, d = self._forward(xs[m], i)
            # merging the results
            ys[m], actions[m], are_exp[m], dists[m, :, :d.size(-1)] = \
                y, a.view(-1), e.view(-1), d.view(d.size(0), d.size(1), -1)
        actions = actions.view(-1)  # flattens the actions tensor, but does not produce a scalar
        assert len(actions) == len(are_exp) == dists.size(0) == len(mxs)
        # amending the metas
        for ia, a, e, d, mx in zip(mask, actions, are_exp, dists.split(1, dim=0), mxs):
            if ia:
                mx.append('actions', a, new_step=True)
                mx.append('is_exploratory', e.squeeze())
                mx.append('states', d)
                mx.append('loss_funcs', self._loss)
                mx.append('reward_func', self.additional_reward_func)
                self.additional_reward_func.register(d, a)
        return ys, mxs, actions

