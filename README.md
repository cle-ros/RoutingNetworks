# Pytorch-Routing
Pytorch-Routing is a pytorch-based implementation of 'RoutingNetworks' for Python 3.5+. The best overview over the work can probably be found in:

Clemens Rosenbaum, Ignacio Cases, Matthew Riemer, Tim Klinger - _Routing Networks and the Challenges of Modular and Compositional Computation_ (arxiv)

https://arxiv.org/abs/1904.12774

The idea was originally published in ICLR:

Clemens Rosenbaum, Tim Klinger, Matthew Riemer - _Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning_ (ICLR 2018).

https://openreview.net/forum?id=ry8dvM-R-

An extension to language domains was introduced in NAACL:

Ignacio Cases, Clemens Rosenbaum, Matthew Riemer, Atticus Geiger, Tim Klinger, Alex Tamkin, Olivia Li, Sandhini Agarwal, Joshua D. Greene, Dan Jurafsky, Christopher Potts and Lauri Karttunen "Recursive Routing Networks: Learning to Compose Modules for Language Understanding" (NAACL 2019).

https://www.aclweb.org/anthology/N19-1365

The latest research on "dispatched" routing networks for single task learning can be found here:

Clemens Rosenbaum, Ignacio Cases, Matthew Riemer, Atticus Geiger, Lauri Karttunen, Joshua D. Greene, Dan Jurafsky, Christopher Potts "Dispatched Routing Networks" (Stanford Tech Report 2019).

https://nlp.stanford.edu/projects/sci/dispatcher.pdf

### What's new
I added implementations of several different new decision making algorithms. In particular, I added reparameterization techniques such as Gumbel/Concrete and RELAX. Additionally, I added some Advantage based RL techniques.

I also added a new module called "prefabs" that includes already defined more or less standard routed layers. For now, it only contains an RNN prefab in form of a routed LSTM. Routing for both i2h and h2h layers can be specified at initialization. 

## Implementation
This package provides an implementation of RoutingNetworks that tries to integrate with Pytorch (https://pytorch.org/) as smoothly as possible by providing RoutingNetwork "layers", each implemented as a `nn.Module`.

(To jump the explanations and go to the examples, see [here](EXAMPLES.md)).

The basic functionality of routing is provided by four different kind of modules:

### PytorchRouting.CoreLayers.Initialization.Initialization
As Routing Networks need to track 'meta-information', we need to work around some Pytorch restrictions by extending what a layer takes as an argument and what it returns. This meta-information consists of the trajectories necessary to later on train Reinforcement Learning based routers, and of the actions used for decisions. Consequently, the information passed from one Pytorch-Routing layer to the next is a triplet of the form `(batch, meta_info_list, actions)`.

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
The constructor arcuments are as follows: `num_selections` defines the number of selections available in the next routed layer; `in_features` defines the dimensionality of one sample when passed into this layer (required to construct function approximators for policies); `num_agents` defines the number of agents available at this layer; `exploration` defines the exploration rate for agents that support it; `policy_storage_type` refers to how the agents' policies are stored, and can be either `approx` or `tabular`; `detach` is a bool and refers to whether or not the gradient flow is cut when passed into the agents's approximators; `approx_hidden_dims` defines the hidden layer dimensions if the agents construct their default policy approximator, an MLP; `approx_module` overrides all other approximator settings, and takes an already instantiated policy approximation module for its agents (which are not limited to MLPs); `additional_reward_function` takes as argument an instance of type `PytorchRouting.RewardFunctions.PerAction.*` and that specifies how per-action rewards should be calculated by the agents. As this reward design may vary per layer, it has to be located here, and not in the final loss function as the other rewards are (see below).

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
module_loss, routing_loss = loss_func(batch_estimates, batch_true, meta_list)
```
To train, we can then simply use backprop on the loss and take an optimization step:
```Python
total_loss = module_loss + routing_loss
total_loss.backward()
opt.step()
```
Additionally, the code allows to have different learning rates for different components - such as for the decision-making networks - using pure Pytorch logic:
```Python
opt_decision = optim.SGD(decision_module.parameters(), lr=decision_learning_rate)
opt_module = optim.SGD([... all other parameters ...], lr=module_learning_rate)
```
## Examples
See [here](EXAMPLES.md).
