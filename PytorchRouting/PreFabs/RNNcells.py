import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PytorchRouting.CoreLayers import Selection
import PytorchRouting.DecisionLayers as ptrdl
from PytorchRouting.RewardFunctions.PerAction import RunningAverageCollaborationReward
from PytorchRouting.RewardFunctions.PerAction.PerActionBaseReward import PerActionBaseReward
from PytorchRouting.RewardFunctions.PerAction import RunningAverageCollaborationReward as RACollRew

from PytorchRouting.Helpers.TorchHelpers import FakeFlatModuleList, Identity


class RoutingRNNCellBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
            self,
            in_features,
            hidden_size,
            num_selections,
            depth_routing,
            routing_agent,
            route_i2h=False,
            route_h2h=True,
            recurrent=False,
            nonlin=F.relu,
            num_agents=1,
            exploration=0.1,
            policy_storage_type='approx',
            detach=True,
            approx_hidden_dims=(),
            additional_reward_func=PerActionBaseReward(),
            **kwargs,
        ):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.hidden_size = hidden_size
        self._projection_size = 4*hidden_size
        self.routing_width = num_selections
        self.routing_depth = depth_routing
        self.recurrent = recurrent
        self.nonlin = nonlin

        assert not (route_i2h and recurrent and in_features != self.hidden_size),\
            'Cannot route i2h recurrently if hidden_dim != in_features (hidden: {}, in: {})'.\
                format(self.hidden_size, in_features)
        assert issubclass(routing_agent, ptrdl.Decision), \
            'Please pass the routing_agent as a class-object of the appropriate type. Reveiced {}'.format(routing_agent)

        # pre-computing the defs of the routed layers
        dimensionality_defs_i2h = [in_features] + [self._projection_size] * depth_routing
        dimensionality_defs_h2h = [self.hidden_size] + [self._projection_size] * depth_routing

        # instantiating the different routing types
        if route_i2h and route_h2h:
            # the decision makers
            self.router_i2h, self.selection_i2h = self._create_routing(
                routing_agent, num_agents, exploration, policy_storage_type, detach, approx_hidden_dims,
                additional_reward_func, dimensionality_defs_i2h)
            self.router_h2h, self.selection_h2h = self._create_routing(
                routing_agent, num_agents, exploration, policy_storage_type, detach, approx_hidden_dims,
                additional_reward_func, dimensionality_defs_h2h)
            self._route = self._route_i2h_h2h
        elif route_i2h:
            # the decision makers
            self.router_i2h, self.selection_i2h = self._create_routing(
                routing_agent, num_agents, exploration, policy_storage_type, detach, approx_hidden_dims,
                additional_reward_func, dimensionality_defs_i2h)
            self.linear_h2h = nn.Linear(self.hidden_size, self._projection_size)
            self._route = self._route_i2h
        elif route_h2h:
            # the decision makers
            self.linear_i2h = nn.Linear(self.in_features, self._projection_size)
            self.router_h2h, self.selection_h2h = self._create_routing(
                routing_agent, num_agents, exploration, policy_storage_type, detach, approx_hidden_dims,
                additional_reward_func, dimensionality_defs_h2h)
            self._route = self._route_h2h
        else:
            raise ValueError('Neither i2h nor h2h routing specified. Please use regular RNNCell instead.')

        self.reset_parameters()

    def _create_routing(self, routing_agent, num_agents, exploration, policy_storage_type, detach, approx_hidden_dims,
                    additional_reward_func, dimensionality_defs):
        list_type = nn.ModuleList if not self.recurrent else FakeFlatModuleList
        effective_width = self.routing_width if not self.recurrent else self.routing_width + 1  # for termination action
        effective_depth = self.routing_depth if not self.recurrent else 1
        base_selection = [] if not self.recurrent else [Identity()]
        router = list_type([
            routing_agent(
                num_selections=effective_width,
                in_features=dimensionality_defs[i],
                num_agents=num_agents,
                exploration=exploration,
                policy_storage_type=policy_storage_type,
                detach=detach,
                approx_hidden_dims=approx_hidden_dims,
                additional_reward_func=additional_reward_func
            ) for i in range(effective_depth)
        ])
        selection = list_type([
            Selection(*(base_selection + [  # need base selection for termination action
                nn.Linear(dimensionality_defs[i], dimensionality_defs[i + 1])
                for _ in range(effective_width)
            ]))
            for i in range(effective_depth)
        ])
        return router, selection

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def _init_hidden(self, input_):
        h = input_.new_zeros((input_.size(0), self.args.d_hidden))
        c = input_.new_zeros((input_.size(0), self.args.d_hidden))
        # h = torch.zeros_like(input_)
        # c = torch.zeros_like(input_)
        return h, c

    @abc.abstractmethod
    def forward(self, x, hidden, metas, task_actions=None, mask=None): pass

    def _route(self, x, h, c, metas, task_actions, mask):
        return torch.Tensor(), torch.Tensor(), []

    def _route_internals(self, input, metas, decisions, selections, task_actions=None, mask=None):
        batch_size = len(metas)
        mask = torch.ones(batch_size, dtype=torch.uint8, device=input.device) if mask is None else mask
        for i in range(self.routing_depth):
            if not any(mask):
                break
            input, metas, actions = decisions[i](input, metas, prior_actions=task_actions, mask=mask)
            mask *= (1 - (actions.squeeze() == 0))
            input, metas, _ = selections[i](input, metas, actions, mask=mask)
            if i < (self.routing_depth - 1):
                input = self.nonlin(input)
        return input, metas

    def _route_i2h(self, x, h, c, metas, task_actions, mask):
        i2h, metas = self._route_internals(x, metas, self.router_i2h, self.selection_i2h, task_actions, mask)
        h2h = self.linear_h2h(h)
        return i2h, h2h, metas

    def _route_h2h(self, x, h, c, metas, task_actions, mask):
        i2h = self.linear_i2h(x)
        h2h, metas = self._route_internals(h, metas, self.router_h2h, self.selection_h2h, task_actions, mask)
        return i2h, h2h, metas

    def _route_i2h_h2h(self, x, h, c, metas, task_actions, mask):
        i2h, metas = self._route_internals(x, metas, self.router_i2h, self.selection_i2h, task_actions, mask)
        h2h, metas = self._route_internals(h, metas, self.router_h2h, self.selection_h2h, task_actions, mask)
        return i2h, h2h, metas


class RoutingLSTMCell(RoutingRNNCellBase):
    def forward(self, x, hidden, metas, task_actions=None, mask=None):
        if hidden is None:
            hidden = self._init_hidden(x)
        h, c = hidden

        # Linear mappings
        i2h_x, h2h_x, metas = self._route(x, h, c, metas, task_actions, mask)
        preact = i2h_x + h2h_x

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        c_hat = preact[:, 3 * self.hidden_size:].tanh()  # input gating
        i_t = gates[:, :self.hidden_size]  # input
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]  # forgetting
        o_t = gates[:, -self.hidden_size:]  # output

        c_t = torch.mul(c, f_t) + torch.mul(i_t, c_hat)
        h_t = torch.mul(o_t, c_t.tanh())
        return (h_t, c_t), metas


class RoutingGRUCell(RoutingRNNCellBase):
    def forward(self, x, hidden, metas):
        raise NotImplementedError()
