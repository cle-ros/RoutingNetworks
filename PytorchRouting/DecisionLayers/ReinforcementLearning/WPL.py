"""
This file defines class REINFORCE.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/7/18
"""
import copy
import torch
import torch.nn.functional as F

from .ActorCritic import ActorCritic


class WPL(ActorCritic):
    """
    Weighted Policy Learner (WPL) Multi-Agent Reinforcement Learning based decision making.
    """

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        grad_est = cum_return - state[:, :, 1].gather(index=action.unsqueeze(1), dim=1).view(-1)
        grad_projected = torch.where(grad_est < 0, 1. + grad_est, 2. - grad_est)
        prob_taken = state[:, :, 0].gather(index=action.unsqueeze(1), dim=1).view(-1)
        prob_target = (prob_taken * grad_projected).detach()
        act_loss = F.mse_loss(prob_taken, prob_target, reduction='none')
        ret_loss = F.mse_loss(state[:, :, 1].gather(index=action.unsqueeze(1), dim=1).view(-1),
                              cum_return.detach(), reduction='none').view(-1)
        return act_loss + ret_loss

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        # policy = F.relu(policy) - F.relu(policy - 1.) + 1e-6
        policy = (policy.transpose(0, 1) - policy.min(dim=1)[0]).transpose(0, 1) + 1e-6
        # policy = policy/policy.sum(dim=1)
        values = self._qvalue_mem[agent](xs)
        distribution = torch.distributions.Categorical(probs=policy)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        state = torch.stack([distribution.logits, values], 2)
        return xs, actions, self._eval_stochastic_are_exp(actions, state), state

    # @staticmethod
    # def _loss(sample):
    #     grad_est = sample.cum_return - sample.state[:, 0, 1]
    #     # ret_loss = F.smooth_l1_loss(sample.state[:, sample.action, 1], sample.cum_return).unsqueeze(-1)
    #     ret_loss = F.smooth_l1_loss(sample.state[:, 0, 1], sample.cum_return).unsqueeze(-1)
    #     grad_projected = grad_est * 1.3
    #     grad_projected = torch.pow(grad_projected, 3.)
    #     if grad_projected < 0:
    #         pol_update = 1. + grad_projected
    #     else:
    #         pol_update = 2. - grad_projected
    #     pol_update = sample.state[:, sample.action, 0] * pol_update
    #     act_loss = F.smooth_l1_loss(sample.state[:, sample.action, 0], pol_update.data)
    #     return act_loss + ret_loss

    # # @staticmethod
    # def _loss(self, sample):
    #     grad_est = sample.cum_return - sample.state[:, 0, 1]
    #     # ret_loss = F.smooth_l1_loss(sample.state[:, sample.action, 1], sample.cum_return).unsqueeze(-1)
    #     # ret_loss = F.smooth_l1_loss(sample.state[:, 0, 1], sample.cum_return).unsqueeze(-1)
    #     grad_projected = grad_est * 1.3
    #     grad_projected = torch.pow(grad_projected, 3.)
    #     if grad_projected < 0:
    #         pol_update = 1. + grad_projected
    #     else:
    #         pol_update = 2. - grad_projected
    #     pol_update = sample.state[:, sample.action, 0] * pol_update
    #     self._policy[sample.prior_action]._approx.data[0, sample.action] = pol_update.data
    #     self._qvalue_mem[sample.prior_action]._approx.data[0, sample.action] = \
    #         0.9 * self._qvalue_mem[sample.prior_action]._approx.data[0, sample.action] + 0.1 * sample.cum_return
    #     # act_loss = F.smooth_l1_loss(sample.state[:, sample.action, 0], pol_update.data)
    #     return torch.zeros(1).to(sample.action.device)
