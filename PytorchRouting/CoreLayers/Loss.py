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

    def __init__(self, pytorch_loss_func, routing_reward_func, discounting=1., clear=False):
        nn.Module.__init__(self)
        self._discounting = discounting
        self._loss_func = pytorch_loss_func
        self._clear = clear
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
            traj_length = len(actions)
            add_rewards = my.add_rewards or [0.] * traj_length
            rewards = []
            for state, action, reward_func, add_r in zip(states, actions, my.reward_func, add_rewards):
                # normalize the per-action reward to the entire sequence length
                per_action_reward = reward_func.get_reward(state, action) / traj_length
                rewards.append(per_action_reward + add_r)
                reward_functions.add(reward_func)
            rewards[-1] += my.final_reward
            returns = [0.]
            for i, rew in enumerate(reversed(rewards)):
                returns.insert(0, rew + returns[0])
            returns = returns[:-1]
            rl_tuples.append([RLSample(lf, s, a, rew, ret, pa, ns, na) for lf, s, a, rew, ret, pa, ns, na in
                          zip(my.loss_funcs, states, actions, rewards, returns,
                              ([None] + actions)[:-1], (states + [None])[1:], (actions + [None])[1:])])
        if self._clear:
            for rf in reward_functions:
                rf.clear()
        return rl_tuples

    def forward(self, ysest, mys, ystrue=None, external_losses=None, reduce=True):
        assert (ystrue is None) != (external_losses is None), 'Must provide either of ystrue or external_losses.'
        if external_losses is not None:
            assert external_losses.size()[0] == len(mys), 'One loss value per sample is required.'
            module_loss = external_losses.view(external_losses.size()[0], -1).sum(dim=1)
            ystrue = torch.zeros_like(module_loss)
        else:
            module_loss = self._loss_func(ysest, ystrue)
            if len(module_loss.size()) > 1:
                module_loss = module_loss.sum(dim=1)
            module_loss = module_loss.unsqueeze(-1)
        assert len(module_loss) == len(mys) == len(ysest) == len(ystrue), \
            'Losses, metas, predictions and targets need to have the same length ({}, {}, {}, {})'.format(
                len(module_loss), len(mys), len(ysest), len(ystrue))
        for l, my, yest, ytrue in zip(module_loss.split(1, dim=0), mys, ysest.split(1, dim=0), ystrue.split(1, dim=0)):
            my.final_reward = self._reward_func(l, yest, ytrue)
        rl_tuples = self._get_rl_tuple_list(mys)
        routing_loss = []
        for traj in rl_tuples:
            traj_loss = 0.
            for sample in traj:
                traj_loss += sample.loss_function(sample).squeeze()
            routing_loss.append(traj_loss)
        routing_loss = torch.stack(routing_loss, dim=0)
        if reduce:
            module_loss = module_loss.mean()
            routing_loss = routing_loss.mean()
        return module_loss, routing_loss
