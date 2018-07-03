### Examples
The following examples can be found in `PytorchRouting.Examples.Models`, and can be tested on CIFAR100-MTL and MNIST-MTL with `PytorchRouting.Examples.run_experiment.py`.

#### Per-Task Agents
The most architectures introduced in the paper assign one agent exclusively to each task. Using Pytorch-Routing, these can be implemented as follows (here, using a WPL MARL agent):
```Python
class RoutedAllFC(nn.Module):
    def __init__(self, in_channels, convnet_out_size, out_dim, num_modules, num_agents):

        self.convolutions = nn.Sequential(
            SimpleConvNetBlock(in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        )
        self._loss_func = Loss(torch.nn.MSELoss(), CorrectClassifiedReward(), discounting=1.)

        self._initialization = Initialization()
        self._per_task_assignment = PerTaskAssignment()

        self._decision_1 = WPL(
            num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
            additional_reward_func=CollaborationReward(reward_ratio=0.3, num_actions=num_modules))
        self._decision_2 = WPL(
            num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
            additional_reward_func=CollaborationReward(reward_ratio=0.3, num_actions=num_modules))
        self._decision_3 = WPL(
            num_modules, convnet_out_size, num_agents=num_agents, policy_storage_type='tabular',
            additional_reward_func=CollaborationReward(reward_ratio=0.3, num_actions=num_modules))

        self._selection_1 = Selection(*[LinearWithRelu(convnet_out_size, 48) for _ in range(num_modules)])
        self._selection_2 = Selection(*[LinearWithRelu(48, 48) for _ in range(num_modules)])
        self._selection_3 = Selection(*[nn.Linear(48, out_dim) for _ in range(num_modules)])
        # self._selection_f = Selection(*[nn.Linear(48, out_dim) for _ in range(num_modules)])

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

    def loss(self, yhat, ytrue, ym):
        return self._lossfunc(yhat, ytrue, ym)
```
Again, the `PerTaskAssignment` layer is utilized to produce actions. However, these actions are not used to select a module, but to select an agent instead. Thus they cannot be overridden, but are explicitly passed to each of the decision making agents as dispatcher-actions.

#### Dispatched Routing Architectures
Extending this paradigm to dispatched architectures is straightforward:
```Python
class DispatchedRoutedAllFC(RoutedAllFC):
    def __init__(self, dispatcher_decision_maker, decision_maker, in_channels, convnet_out_size,
                 out_dim, num_modules, num_agents):
        RoutedAllFC.__init__(self, decision_maker, in_channels, convnet_out_size, out_dim, num_modules, num_agents)
        self._per_task_assignment = dispatcher_decision_maker(
            num_agents, convnet_out_size, num_agents=1, policy_storage_type='approx',
            additional_reward_func=CollaborationReward(reward_ratio=0.0, num_actions=num_modules))
```
Here, the task-specific assignment "agent" simply got replaced by a separate dispatching agent.