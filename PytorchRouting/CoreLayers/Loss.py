"""
This file defines class RoutingLoss.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""
import abc
from collections import defaultdict

import torch
import torch.nn as nn

from PytorchRouting.RewardFunctions.Final import CorrectClassifiedReward


class Loss(nn.Module, metaclass=abc.ABCMeta):
    """
    This function defines the combined module/decision loss functions. It performs four steps that will result in
    separate losses for the modules and the decision makers:
    1. it computes the module losses
    2. it translates these module losses into per-sample reinforcement learning rewards
    3. it uses these final rewards to compute the full rl-trajectories for each sample
    4. it uses the decision-making specific loss functions to compute the total decision making loss
    """

    def __init__(self, pytorch_loss_func, routing_reward_func, discounting=1., clear=False,
                 normalize_per_action_rewards=True):
        nn.Module.__init__(self)
        self._discounting = discounting
        self._loss_func = pytorch_loss_func
        self._clear = clear
        try:
            self._loss_func.reduction = 'none'
        except AttributeError:
            pass
        self._reward_func = routing_reward_func
        self._npar = normalize_per_action_rewards

    def _get_rl_loss_tuple_map(self, mys, device):
        rl_loss_tuple_map = defaultdict(lambda: defaultdict(list))
        reward_functions = set()
        for traj_counter, my in zip(torch.arange(len(mys), device=device).unsqueeze(1), mys):
            my.finalize()  # translates the trajectory from a list of obj into lists
            my.add_rewards = [ar if ar is not None else 0. for ar in my.add_rewards] \
                if hasattr(my, 'add_rewards') else [0.] * len(my.actions)
            assert len(my.actions) == len(my.states) == len(my.add_rewards) == len(my.reward_func)
            rewards = []
            # computing the rewards
            for state, action, reward_func, add_r in zip(my.states, my.actions, my.reward_func, my.add_rewards):
                # normalize the per-action reward to the entire sequence length
                per_action_reward = (reward_func.get_reward(state, action) + add_r) / len(my.actions)
                # normalize to the final reward, s.t. it will be interpreted as a fraction thereof
                per_action_reward = per_action_reward * torch.abs(my.final_reward) if self._npar else per_action_reward
                rewards.append(per_action_reward)
                reward_functions.add(reward_func)
            rewards[-1] += my.final_reward
            returns = [0.]
            # computing the returns
            for i, rew in enumerate(reversed(rewards)):
                returns.insert(0, rew + returns[0])
            returns = returns[:-1]
            # creating the tensors to compute the loss from the SARSA tuple
            for lf, s, a, rew, ret, pa, ns, na in zip(my.loss_funcs, my.states, my.actions, rewards, returns,
                                                      ([None] + my.actions)[:-1],
                                                      (my.states + [None])[1:],
                                                      (my.actions + [None])[1:]):
                is_terminal = ns is None or s.numel() != ns.numel()
                rl_loss_tuple_map[lf]['indices'].append(traj_counter)
                rl_loss_tuple_map[lf]['is_terminal'].append(torch.tensor([is_terminal], dtype=torch.uint8, device=device))
                rl_loss_tuple_map[lf]['states'].append(s)
                rl_loss_tuple_map[lf]['actions'].append(a.view(-1))
                rl_loss_tuple_map[lf]['rewards'].append(rew.view(-1))
                rl_loss_tuple_map[lf]['returns'].append(ret.view(-1))
                rl_loss_tuple_map[lf]['final_reward'].append(my.final_reward.view(-1))
                rl_loss_tuple_map[lf]['prev_actions'].append(a.new_zeros(1) if pa is None else pa.view(-1))
                rl_loss_tuple_map[lf]['next_states'].append(s if is_terminal else ns)
                rl_loss_tuple_map[lf]['next_actions'].append(a.new_zeros(1) if is_terminal else na.view(-1))
        # concatenating the retrieved values into tensors
        for k0, v0 in rl_loss_tuple_map.items():
            for k1, v1 in v0.items():
                v0[k1] = torch.cat(v1, dim=0)
        if self._clear:
            for rf in reward_functions:
                rf.clear()
        return rl_loss_tuple_map

    def forward(self, ysest, mys, ystrue=None, external_losses=None, reduce=True):
        assert not(ystrue is None and external_losses is None), \
            'Must provide ystrue and possibly external_losses (or both).'
        batch_size = ysest.size(0)
        if external_losses is not None:
            # first case: external losses are provided externally
            assert external_losses.size()[0] == len(mys), 'One loss value per sample is required.'
            module_loss = external_losses.view(external_losses.size()[0], -1).sum(dim=1)
        else:
            # second case: they are not, so we need to compute them
            module_loss = self._loss_func(ysest, ystrue)
            if len(module_loss.size()) > 1:
                module_loss = module_loss.sum(dim=1).reshape(-1)
        if ystrue is None:
            # more input checking
            assert not isinstance(self._reward_func, CorrectClassifiedReward), \
                'Must provide ystrue when using CorrectClassifiedReward'
            ystrue = ysest.new_zeros(batch_size)
        assert len(module_loss) == len(mys) == len(ysest) == len(ystrue), \
            'Losses, metas, predictions and targets need to have the same length ({}, {}, {}, {})'.format(
                len(module_loss), len(mys), len(ysest), len(ystrue))
        # add the final reward, as we can only compute them now that we have the external feedback
        for l, my, yest, ytrue in zip(module_loss.split(1, dim=0), mys, ysest.split(1, dim=0), ystrue.split(1, dim=0)):
            my.final_reward = self._reward_func(l, yest, ytrue)
        # retrieve the SARSA pairs to compute the respective decision making losses
        rl_loss_tuple_map = self._get_rl_loss_tuple_map(mys, device=ysest.device)
        # initialize the rl loss
        routing_loss = torch.zeros(batch_size, dtype=torch.float, device=ysest.device)
        for loss_func, rl_dict in rl_loss_tuple_map.items():
            # batch the RL loss by loss function, if possible
            rl_losses = loss_func(rl_dict['is_terminal'], rl_dict['states'], rl_dict['next_states'], rl_dict['actions'],
                                  rl_dict['next_actions'], rl_dict['rewards'], rl_dict['returns'], rl_dict['final_reward'])
            for i in torch.arange(batch_size, device=ysest.device):
                # map the losses back onto the sample indices
                routing_loss[i] = routing_loss[i] + torch.sum(rl_losses[rl_dict['indices'] == i])
        if reduce:
            module_loss = module_loss.mean()
            routing_loss = routing_loss.mean()
        return module_loss, routing_loss
