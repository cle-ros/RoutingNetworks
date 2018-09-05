"""
This file defines class Models.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/14/18
"""
try:
    import cPickle as pickle
except ImportError:
    import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from PytorchRouting.CoreLayers import Initialization, Loss, Selection
from PytorchRouting.DecisionLayers import PerTaskAssignment
from PytorchRouting.DecisionLayers.Decision import Decision
from PytorchRouting.RewardFunctions.Final import CorrectClassifiedReward, NegLossReward
from PytorchRouting.RewardFunctions.PerAction import CollaborationReward


class SimpleConvNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.maxpool(y)
        return y


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)


class LinearWithRelu(nn.Linear):
    def forward(self, input):
        output = nn.Linear.forward(self, input)
        output = F.relu(output)
        return output


class RoutedAllFC(nn.Module):
    """
    This is a non-optimized example model which can be used as a starting point for MTL experiments.
    """
    def __init__(self, decision_maker, in_channels, convnet_out_size, out_dim, num_modules, num_agents):
        nn.Module.__init__(self)
        self.convolutions = nn.Sequential(
            SimpleConvNetBlock(in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        )
        # self._loss_func = Loss(torch.nn.CrossEntropyLoss(), NegLossReward(), discounting=1.)
        self._loss_func = Loss(torch.nn.MSELoss(), CorrectClassifiedReward(), discounting=1.)

        self._initialization = Initialization()
        self._per_task_assignment = PerTaskAssignment()

        self._decision_1 = decision_maker(
            num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
            additional_reward_func=CollaborationReward(reward_ratio=0.1, num_actions=num_modules))
        self._decision_2 = decision_maker(
            num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
            additional_reward_func=CollaborationReward(reward_ratio=0.1, num_actions=num_modules))
        self._decision_3 = decision_maker(
            num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
            additional_reward_func=CollaborationReward(reward_ratio=0.1, num_actions=num_modules))

        self._selection_1 = Selection(*[LinearWithRelu(convnet_out_size, 64) for _ in range(num_modules)])
        self._selection_2 = Selection(*[LinearWithRelu(64, 64) for _ in range(num_modules)])
        self._selection_3 = Selection(*[nn.Linear(64, out_dim) for _ in range(num_modules)])

    def forward(self, x, tasks):
        y = self.convolutions(x)
        y, meta, actions = self._initialization(y, tasks=tasks)
        y, meta, task_actions = self._per_task_assignment(y, meta, actions)
        y, meta, routing_actions_1 = self._decision_1(y, meta, task_actions)
        y, meta, _ = self._selection_1(y, meta, routing_actions_1)
        y, meta, routing_actions_2 = self._decision_2(y, meta, task_actions)
        y, meta, _ = self._selection_2(y, meta, routing_actions_2)
        y, meta, routing_actions_3 = self._decision_3(y, meta, task_actions)
        y, meta, _ = self._selection_3(y, meta, routing_actions_3)
        return y, meta

    def _get_params_by_class(self, cls):
        params = []
        for mod in self.modules():
            if mod is self:
                continue
            if isinstance(mod, cls):
                params += list(mod.parameters())
        return params

    def routing_parameters(self):
        return self._get_params_by_class(Decision)

    def module_parameters(self):
        params = self._get_params_by_class(Selection)
        params += list(self.convolutions.parameters())
        return params

    def loss(self, yhat, ytrue, ym):
        return self._loss_func(yhat, ytrue, ym)

    def start_logging_selections(self):
        for m in self.modules():
            if isinstance(m, Selection):
                m.start_logging_selections()

    def stop_logging_selections_and_report(self):
        modules_used = ''
        for m in self.modules():
            if isinstance(m, Selection):
                selections = m.stop_logging_and_get_selections()
                if len(selections) > 0:
                    modules_used += '{}, '.format(len(selections))
        print('        Modules used: {}'.format(modules_used))


class DispatchedRoutedAllFC(RoutedAllFC):
    def __init__(self, dispatcher_decision_maker, decision_maker, in_channels, convnet_out_size,
                 out_dim, num_modules, num_agents):
        RoutedAllFC.__init__(self, decision_maker, in_channels, convnet_out_size, out_dim, num_modules, num_agents)
        self._per_task_assignment = dispatcher_decision_maker(
            num_agents, convnet_out_size, num_agents=1, policy_storage_type='approx',
            additional_reward_func=CollaborationReward(reward_ratio=0.0, num_actions=num_modules))

