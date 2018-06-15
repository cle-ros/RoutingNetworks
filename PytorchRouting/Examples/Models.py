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

from PytorchRouting.UtilLayers import Sequential

from PytorchRouting.CoreLayers import Initialization, Loss, Selection
from PytorchRouting.DecisionLayers import REINFORCE, QLearning, SARSA, ActorCritic, GumbelSoftmax, PerTaskAssignment, \
    WPL
from PytorchRouting.RewardFunctions.Final import CorrectClassifiedReward
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


class PerTask_all_fc(nn.Module):
    def __init__(self, in_channels, convnet_out_size, out_dim, num_modules, num_agents,):
        nn.Module.__init__(self)
        self.convolutions = nn.Sequential(
            SimpleConvNetBlock(in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        )
        self._loss_layer = Loss(torch.nn.CrossEntropyLoss(), CorrectClassifiedReward(), discounting=1.)
        self.fc_layers = Sequential(
            PerTaskAssignment(),
            Selection(*[nn.Linear(convnet_out_size, 64) for _ in range(num_modules)]),
            Selection(*[nn.Linear(64, 64) for _ in range(num_modules)]),
            Selection(*[nn.Linear(64, out_dim) for _ in range(num_modules)]),
        )

    def forward(self, x, tasks):
        y = self.convolutions(x)
        y, meta = self.fc_layers(y, tasks=tasks)
        return y, meta

    def loss(self, yhat, ytrue, ym):
        return self._loss_layer(yhat, ytrue, ym)


class WPL_routed_all_fc(PerTask_all_fc):
    def __init__(self, in_channels, convnet_out_size, out_dim, num_modules, num_agents):
        PerTask_all_fc.__init__(self, in_channels, convnet_out_size, out_dim, num_modules, num_agents)

        self._initialization = Initialization()
        self._per_task_assignment = PerTaskAssignment()

        self._decision_1 = WPL(num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
                               additional_reward_func=CollaborationReward(reward_ratio=0.3, num_actions=num_modules))
        self._decision_2 = WPL(num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
                               additional_reward_func=CollaborationReward(reward_ratio=0.3, num_actions=num_modules))
        self._decision_3 = WPL(num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
                               additional_reward_func=CollaborationReward(reward_ratio=0.3, num_actions=num_modules))

        self._selection_1 = Selection(*[nn.Linear(convnet_out_size, 64) for _ in range(num_modules)])
        self._selection_2 = Selection(*[nn.Linear(64, 64) for _ in range(num_modules)])
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
