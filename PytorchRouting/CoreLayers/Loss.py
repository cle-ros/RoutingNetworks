"""
This file defines class RoutingLoss.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import abc
import torch
import torch.nn as nn

from PytorchRouting.Helpers.RLSample import RLSample


class Loss(nn.Module, metaclass=abc.ABCMeta):
    """
    Class RoutingLoss defines ...
    """

    def __init__(self, pytorch_loss_func, routing_reward_func):
        nn.Module.__init__(self)
        self._loss_func = pytorch_loss_func
        self._reward_func = routing_reward_func

    # def __getattr__(self, item):
    #     return getattr(self._loss_func, item)

    def _get_rl_tuple_list(self, mys):
        rl_tuples = []
        reward_functions = set()
        for my in mys:
            actions = my.actions
            states = my.states
            rewards = []
            for state, action, reward_func in zip(states, actions, my.reward_func):
                rewards.append(reward_func.get_reward(state, action))
                reward_functions.add(reward_func)
            rewards.append(my.final_reward)
            returns = [0.]
            for i, rew in enumerate(reversed(rewards)):
                returns.append(rew + returns[-1] * (self._discounting ** (i - 1)))
            returns = torch.cumsum(torch.cat(list(reversed(returns[1:])), dim=0), 0)[:-1]
            rl_tuples += [RLSample(lf, s, a, rew, ret, ns, na) for lf, s, a, rew, ret, ns, na in
                          zip(my.loss_funcs, states, actions, rewards, returns,
                              (states + [None])[1:], (actions + [None])[1:])
                          ]
        for rf in reward_functions:
            rf.clear()
        return rl_tuples

    def forward(self, ysest, ystrue, mys):
        module_loss = self._loss_func(ysest, ystrue).unsqueeze(-1)
        for l, my in zip(module_loss.split(1, dim=0), mys):
            my.final_reward = self._reward_func(l)
        rl_tuples = self._get_rl_tuple_list(mys)
        routing_loss = 0.
        for sample in rl_tuples:
            routing_loss += sample.loss_function(sample)
        return module_loss, routing_loss
