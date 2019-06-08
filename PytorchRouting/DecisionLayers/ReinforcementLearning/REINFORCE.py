"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import torch
import torch.nn.functional as F

from ..Decision import Decision


class REINFORCE(Decision):
    """
    REINFORCE (likelihood ratio policy gradient) based decision making.
    """

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        return - state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1) * cum_return.detach()

    def set_exploration(self, exploration):
        if self._set_pg_temp:
            self._pg_temperature = 0.1 + 9.9*exploration

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy/self._pg_temperature)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        return xs, actions, self._eval_stochastic_are_exp(actions, distribution.logits), distribution.logits


class REINFORCEBl1(Decision):
    def __init__(self, *args, value_net=None, **kwargs):
        Decision.__init__(self, *args, **kwargs)
        self._value_mem = self._construct_policy_storage(
            self._num_selections, self._pol_type, value_net, self._pol_hidden_dims)

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        normalized_return = (cum_return - state[:, 0, 1].view(-1)).detach()
        act_loss = - state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1) * normalized_return
        value_target = torch.where(is_terminal, final_reward, next_state[:, :, 1].max(dim=1)[0] - reward).detach()
        val_loss = F.mse_loss(state[:, :, 1].gather(index=action.unsqueeze(1), dim=1).view(-1),
                              value_target, reduction='none').view(-1)
        return act_loss + val_loss

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        values = self._value_mem[agent](xs).expand_as(policy)
        distribution = torch.distributions.Categorical(logits=policy)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        state = torch.stack([distribution.logits, values], 2)
        return xs, actions, self._eval_stochastic_are_exp(actions, state), state


class REINFORCEBl2(REINFORCEBl1):
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        logits = state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1)
        probs = F.softmax(state[:, :, 0], dim=1).gather(index=action.unsqueeze(1), dim=1).view(-1)
        normalized_return = (cum_return/(self._num_selections * probs) - state[:, 0, 1].view(-1)).detach()
        act_loss = - logits * normalized_return
        value_target = torch.where(is_terminal, final_reward, next_state[:, :, 1].max(dim=1)[0] - reward).detach()
        val_loss = F.mse_loss(state[:, :, 1].gather(index=action.unsqueeze(1), dim=1).view(-1),
                              value_target, reduction='none').view(-1)
        return act_loss + val_loss


class EGreedyREINFORCE(REINFORCE):

    def set_exploration(self, exploration):
        # because of the special nature of this approach, exploration needs to be calculated differently
        self._exploration = min(1., 3.*exploration)

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        importance_weights = (state[:, :, 1] / state[:, :, 2]).gather(1, action.unsqueeze(1))
        importance_weighted_return = (importance_weights * cum_return).detach()
        return - state[:, :, 0].gather(dim=1, index=action.unsqueeze(1)).view(-1) * importance_weighted_return

    def _forward(self, xs, agent):
        batch_dim = xs.size(0)
        policy = self._policy[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy)
        if self.training:
            exploration_dist = torch.ones(batch_dim, 2).float()
            exploration_dist[:, 0] *= 1 - self._exploration
            exploration_dist[:, 1] *= self._exploration
            explore_bin = torch.multinomial(exploration_dist, 1).byte().to(xs.device)
            selected_probs, greedy = distribution.logits.max(dim=1)
            on_policy = distribution.sample()
            actions = torch.where(explore_bin, on_policy.unsqueeze(-1), greedy.unsqueeze(-1))

            # computing the importance weights
            sampling_dist = distribution.probs.detach() * \
                            (1 / (1 - selected_probs.unsqueeze(1).expand_as(policy))) * \
                            (self._exploration / (policy.size(1) - 1))
            sampling_dist.scatter_(1,
                                   greedy.unsqueeze(1),
                                   torch.ones(batch_dim, 1, device=xs.device) * (1 - self._exploration))
            state = torch.stack((distribution.logits, distribution.probs, sampling_dist), dim=2)
        else:
            actions = distribution.logits.max(dim=1)[1]
            state = torch.stack((distribution.logits, distribution.probs, distribution.probs), dim=2)
        return xs, actions, self._eval_stochastic_are_exp(actions, distribution.logits), state
