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
    This function defines the combined module/decision loss functions. It performs four steps that will result in
    separate losses for the modules and the decision makers:
    1. it computes the module losses
    2. it translates these module losses into per-sample reinforcement learning rewards
    3. it uses these final rewards to compute the full rl-trajectories for each sample
    4. it uses the decision-making specific loss functions to compute the total decision making loss
    """

    def __init__(self, pytorch_loss_func, routing_reward_func, discounting=1.):
        nn.Module.__init__(self)
        self._discounting = discounting
        self._loss_func = pytorch_loss_func
        try:
            self._loss_func.reduce = False
        except AttributeError:
            pass
        self._reward_func = routing_reward_func

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
            rewards.append(my.final_reward.unsqueeze(0))
            returns = [0.]
            for i, rew in enumerate(reversed(rewards)):
                returns.append(rew + returns[-1] * (self._discounting ** (i - 1)))
            returns = torch.cumsum(torch.stack(list(reversed(returns[1:])), dim=0), 0)[:-1]
            rl_tuples += [RLSample(lf, s, a, rew, ret, ns, na) for lf, s, a, rew, ret, ns, na in
                          zip(my.loss_funcs, states, actions, rewards, returns,
                              (states + [None])[1:], (actions + [None])[1:])
                          ]
        for rf in reward_functions:
            rf.clear()
        return rl_tuples

    def forward(self, ysest, ystrue, mys):
        module_loss = self._loss_func(ysest, ystrue.squeeze()).view(len(mys), -1).sum(dim=1).unsqueeze(-1)
        for l, my, yest, ytrue in zip(module_loss.split(1, dim=0), mys, ysest.split(1, dim=0), ystrue.split(1, dim=0)):
            my.final_reward = self._reward_func(l, yest, ytrue)
        module_loss = torch.sum(module_loss)
        rl_tuples = self._get_rl_tuple_list(mys)
        routing_loss = 0.
        for sample in rl_tuples:
            routing_loss += sample.loss_function(sample).squeeze()
        return module_loss, routing_loss
