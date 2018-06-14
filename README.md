# Pytorch-Routing
Pytorch-Routing is a pytorch-based implementation of 'RoutingNetworks':

Clemens Rosenbaum, Tim Klinger, Matthew Riemer - _Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning_ (ICLR 2018).

https://openreview.net/forum?id=ry8dvM-R-

Multi-task learning (MTL) with neural networks leverages commonalities in tasks to improve performance, but often suffers from task interference which reduces the benefits of transfer. To address this issue we introduce the routing network paradigm, a novel neural network and training algorithm. A routing network is a kind of self-organizing neural network consisting of two components: a router and a set of one or more function blocks. A function block may be any neural network – for example a fully-connected or a convolutional layer. Given an input the router makes a routing decision, choosing a function block to apply and passing the output back to the router recursively, terminating when a fixed recursion depth is reached. In this way the routing network dynamically composes different function blocks for each input. We employ a collaborative multi-agent reinforcement learning (MARL) approach to jointly train the router and function blocks. We evaluate our model against cross-stitch networks and shared-layer baselines on multi-task settings of the MNIST, mini-imagenet, and CIFAR-100 datasets. Our experiments demonstrate a significant improvement in accuracy, with sharper convergence. In addition, routing networks have nearly constant per-task training cost while cross-stitch networks scale linearly with the number of tasks. On CIFAR100 (20 tasks) we obtain cross-stitch performance levels with an 85% average reduction in training time.

## Implementation
This package provides an implementation of RoutingNetworks that tries to integrate with Pytorch (https://pytorch.org/) by providing RoutingNetwork "layers", each implemented as a `nn.Module`.

The basic functionality of routing is provided by four different kind of modules:

### PytorchRouting.CoreLayers.Initialization.Initialization
One of the problems of implementing RoutingNetworks with pytorch is that we need to track 'meta-information'. This meta-information consists of the trajectories necessary to later on train Reinforcement Learning based routers, and of the actions used for decisions. Consequently, the information passed from one Pytorch-Routing layer to the next is a triplet of the form `(batch, meta_info_list, actions)`.

The initialization of the meta-information objects - one for each sample in a batch - is thus the first required step when using this package, and is achieved with the `Initialization` module.
```Python
init = Initialization()
batch, meta_list, actions = init(batch, tasks=())
```
The initialization module takes the batch - in form of a Pytorch `Variable` (with the first dim as the batch dim) and an optional list of task-labels (for multi-task learning) and returns the required triplet `(batch, meta_info_list, actions)` (though with empty actions).

### PytorchRouting.Decision.*
The next step in routing a network is to make a routing decision (i.e. creating a selection) for each sample. These layers - with one class for each decision making technique - take the Pytorch-Routing triplet, and make a decision for each sample in the batch. These decisions are logged in the meta-information objects, and returned as a `torch.LongTensor` as the third element of the Pytorch-Routing triplet:

```Python
decision = Decision(
        num_selections,
        in_features,
        num_agents=1,
        exploration=0.1,
        policy_storage_type='approx',
        detach=True,
        approx_hidden_dims=(),
        approx_module=None,
        additional_reward_class=PerActionBaseReward,
        additional_reward_args={})
batch, meta_list, new_actions = decision(batch, meta_list, actions)
```
The constructor arcuments are as follows: `num_selections` defines the number of selections available in the next routed layer; `in_features` defines the dimensionality of one sample when passed into this layer (required to construct function approximators for policies); `num_agents` defines the number of agents available at this layer; `exploration` defines the exploration rate for agents that support it; `policy_storage_type` refers to how the agents' policies are stored, and can be either `approx` or `tabular`; `detach` is a bool and refers to whether or not the gradient flow is cut when passed into the agents's approximators; `approx_hidden_dims` defines the hidden layer dimensions if the agents construct their default policy approximator, an MLP; `approx_module` overrides all other approximator settings, and takes an already instantiated policy approximation module for its agents (which are not limited to MLPs); `additional_reward_function` takes as argument an instance of type `PytorchRouting.RewardFunctions.PerAction.*` and that specifies how per-action rewards should be calculated by the agents.

#### _Dispatching_
The `actions` argument to the layer call will be interpreted as the dispatcher actions specifying the agents to be selected:
```Python
# 1. getting the dispatcher actions
batch, meta_list, dispatcher_actions = decision_dispatcher(batch, meta_list, [])
# 2. passing the dispatcher actions to an agent
batch, meta_list, selection_actions = decision_selector(batch, meta_list, dispatcher_actions)
# 3. selecting the modules (see below)
```
Using a special decision module, this can also be used to implement per-task agents:
```Python
# 1. getting the per-task assignment actions
batch, meta_list, per_task_actions = PytorchRouting.DecisionLayers.Others.PerTaskAssignment()(batch, meta_list, [])
# 2. passing the task assignment preselections
batch, meta_list, selection_actions = decision_selector(batch, meta_list, per_task_actions)
# 3. selecting the modules (see below)
```
### PytorchRouting.CoreLayers.Selection
Now that the actions have been computed, the actual selection of the function block is the next step. This functionality is provided by the `Selection` module:
```Python
selection = Selection(*modules)
batch_out, meta_list, actions = selection(batch, meta_list, actions)
```
Once the module has been initialized by passing in a list of initialized modules, it's application is straightforward. An example of how to initialize the selection layer can look as follows:
```Python
# for 5 different fully connected layers with the same number of parameters
selection = Selection(*[nn.Linear(in_dim, out_dim) for _ in range(5)])
# for 2 different MLP's, with different number of parameters.
selection = Selection(MLP(in_dim, out_dim, hidden=(64, 128)), MLP(in_dim, out_dim, hidden=(64, 64)))
```

### PytorchRouting.CoreLayers.Loss
The final function is a Pytorch-Routing specific loss module. This is required as the loss from the normal training needs to be translated (per-sample) to a Reinforcement Learning reward signal:
```Python
loss_func = Loss(pytorch_loss_func, routing_reward_func)
module_loss, routing_loss = loss_func(batch_estimates, batch_true, meta_list)
```
The loss module is instantiated by passing in two different other modules - a pytorch loss function (i.e. a `nn.*Loss*` module) and a reward function (from `PytorchRouting.RewardFunctions.Final.*`) to translate to a reward. Once instantiated, it takes different arguments than the other "layer-like" modules of Pytorch-Routing. These arguments are the batch estimates, i.e. the first output of the routing-triplet, the true targets and the meta-list, i.e. the second output of the routing-triplet. An example could be:
```Python
loss_func = Loss(torch.nn.CrossEntropyLoss(), NegLossReward())
```
To train, we can then simply use backprop on the loss and take an optimization step:
```Python
module_loss, routing_loss = loss_func(batch_estimates, batch_true, meta_list)
total_loss = module_loss + routing_loss
total_loss.backward()
opt.step()
```
Additionally, the code allows to have different learning rates for different components - such as for the decision-making networks - using pure Pytorch logic:
```Python
opt_decision = optim.SGD(decision_module.parameters, lr=decision_learning_rate)
opt_module = optim.SGD([... all other modules ...], lr=module_learning_rate)
```