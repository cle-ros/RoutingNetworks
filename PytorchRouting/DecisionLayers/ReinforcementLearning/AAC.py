"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import copy
import torch
import torch.nn.functional as F

from ..Decision import Decision


class AAC(Decision):
    """
    ActorCritic based decision making.
    """
    def __init__(self, *args, value_net=None, **kwargs):
        Decision.__init__(self, *args, **kwargs)
        self._value_mem = self._construct_policy_storage(
            1, self._pol_type, value_net, self._pol_hidden_dims)

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        advantages = (cum_return - state[:, 0, 1])
        act_loss = - state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1) * advantages.detach()
        val_loss = advantages.pow(2).mean()
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


class BootstrapAAC(AAC):
    """
    ActorCritic based decision making.
    """
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        next_state = torch.where(is_terminal, torch.zeros_like(next_state, device=state.device), next_state)
        advantages = (next_state[:, 0, 1] + reward - state[:, 0, 1])
        act_loss = - state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1) * advantages.detach()
        val_loss = advantages.pow(2).mean()
        return act_loss + val_loss


class EGreedyAAC(AAC):
    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        importance_weights = (state[:, :, 1] / state[:, :, 2]).gather(1, action.unsqueeze(1))
        advantages = (cum_return.detach() - state[:, 0, 3])
        importance_weighted_advantages = (importance_weights * advantages).detach()
        act_loss = - state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1) * importance_weighted_advantages
        val_loss = advantages.pow(2).mean()
        return act_loss + val_loss

    def _forward(self, xs, agent):
        batch_dim = xs.size(0)
        policy = self._policy[agent](xs)
        values = self._value_mem[agent](xs).expand_as(policy)
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
            state = torch.stack((distribution.logits, distribution.probs, sampling_dist, values), dim=2)
        else:
            actions = distribution.logits.max(dim=1)[1]
            state = torch.stack((distribution.logits, distribution.probs, distribution.probs, values), dim=2)
        return xs, actions, self._eval_stochastic_are_exp(actions, state), state
