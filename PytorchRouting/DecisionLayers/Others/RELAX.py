"""
This file defines class REBAR. Implementation largely taken from:
    https://github.com/duvenaud/relax/blob/master/pytorch_toy.py

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/12/18
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad as compute_gradient

from ..Decision import Decision


def entropy(probs):
    return torch.sum(- torch.log(probs) * probs, dim=1)


def make_samples(logits, temp, eps=1e-8):
    u1 = torch.zeros(*logits.shape, device=logits.device)
    u2 = torch.zeros(*logits.shape, device=logits.device)
    u1.uniform_()
    u2.uniform_()
    # temp = tf.exp(log_temp)
    # logprobs = tf.nn.log_softmax(logits)
    logprobs = torch.distributions.Categorical(logits=logits).logits
    g = -torch.log(-torch.log(u1 + eps) + eps)
    scores = logprobs_z = logprobs + g
    hard_samples = torch.argmax(scores, dim=1)
    # hard_samples_oh = tf.one_hot(hard_samples, scores.get_shape().as_list()[1])
    hard_samples_onehot = torch.zeros(hard_samples.size(0), scores.size(1), device=logits.device)
    hard_samples_onehot.scatter_(1, hard_samples.unsqueeze(1), 1)

    g2 = -torch.log(-torch.log(u2 + eps) + eps)
    scores2 = logprobs + g2

    # B = tf.reduce_sum(scores2 * hard_samples_oh, axis=1, keep_dims=True) - logprobs
    B = scores2 * hard_samples_onehot - logprobs
    y = -1. * torch.log(u2) + torch.exp(-1. * B)
    g3 = -1. * torch.log(y)
    scores3 = g3 + logprobs
    # slightly biased…
    logprobs_zt = hard_samples_onehot * scores2 + ((-1. * hard_samples_onehot) + 1.) * scores3
    return hard_samples, F.softmax(logprobs_z / temp, dim=1), F.softmax(logprobs_zt / temp, dim=1)


class RELAX(Decision):
    """
    Class GumbelSoftmax defines a decision making procedure that uses the GumbelSoftmax reparameterization trick
    to perform differentiable sampling from the categorical distribution.
    """
    def __init__(self, *args, value_net=None, **kwargs):
        Decision.__init__(self, *args, **kwargs)
        # translating exploration into the sampling temperature parameter in [0.1, 10]

        self._value_mem = self._construct_policy_storage(
            1, self._pol_type, value_net, self._pol_hidden_dims, in_dim=self._in_features + self._num_selections)
        self._temperature = 0.5
        self._value_coefficient = 0.5
        self._entropy_coefficient = 0.01

    def set_exploration(self, exploration):
        self._temperature = 0.1 + 9.9*exploration

    def _loss(self, is_terminal, state, next_state, action, next_action, reward, cum_return, final_reward):
        if not self.training:
            # we cannot compute gradients in test mode, so we cannot compute the respective losses
            return torch.zeros(state.size(0), device=state.device)
        # oh_A = tf.one_hot(train_model.a0, ac_space.n)
        onehot_action = torch.zeros(state.size(0), state.size(1), device=state.device)
        onehot_action.scatter_(1, action.unsqueeze(1), 1)

        log_policy = state[:, :, 0]
        values = state[:, :, 1]
        values_t = state[:, :, 2]
        policy = F.softmax(log_policy, dim=1)
        policy_entropy = entropy(policy)

        # params = find_trainable_variables("model")
        params = self.parameters()
        # policy_params = [v for v in params if "pi" in v.name]
        policy_params = list(self._policy.parameters())
        # vf_params = [v for v in params if "vf" in v.name]
        vf_params = list(self._value_mem.parameters())
        # entropy_grads = tf.gradients(entropy, policy_params)
        entropy_grads = compute_gradient(policy_entropy.sum(), policy_params, retain_graph=True, allow_unused=True)


        # ddiff_loss = tf.reduce_sum(train_model.vf - train_model.vf_t)
        # ddiff_grads = tf.gradients(ddiff_loss, policy_params)

        ddiff_loss = (values - values_t).sum()
        ddiff_grads = compute_gradient(ddiff_loss, policy_params,
                                       retain_graph=True, only_inputs=True, create_graph=True, allow_unused=True)

        # sm = tf.nn.softmax(train_model.pi)
        dlogp_dpi = onehot_action * (1. - policy) + (1. - onehot_action) * (-policy)
        # pi_grads = -((tf.expand_dims(R, 1) - train_model.vf_t) * dlogp_dpi)
        pi_grads = -((cum_return.unsqueeze(1).expand_as(values_t) - values_t) * dlogp_dpi)
        # pg_grads = tf.gradients(train_model.pi, policy_params, grad_ys=pi_grads)
        pg_grads = compute_gradient(policy, policy_params, grad_outputs=pi_grads,
                                    retain_graph=True, create_graph=True, allow_unused=True)
        pg_grads = [pg - dg for pg, dg in zip(pg_grads, ddiff_grads) if pg is not None]

        # cv_grads = tf.concat([tf.reshape(p, [-1]) for p in pg_grads], 0)
        cv_grads = torch.cat([p.view(-1) for p in pg_grads], 0)
        cv_grad_splits = torch.pow(cv_grads, 2).sum()
        vf_loss = cv_grad_splits * self._value_coefficient


        for e_grad, p_grad, param in zip(entropy_grads, pg_grads, policy_params):
            if p_grad is None and e_grad is None:
                continue
            elif p_grad is None:
                p_grad = torch.zeros_like(e_grad)
            elif e_grad is None:
                e_grad = torch.zeros_like(p_grad)
            grad = -e_grad * self._entropy_coefficient + p_grad
            if param.grad is not None:
                param.grad.add_(grad)
            else:
                grad = grad.detach()
                grad.requires_grad_(False)
                param.grad = grad

        # cv_grads = compute_gradient(vf_loss, vf_params)
        # for cv_grad, param in zip(cv_grads, vf_params):
        #     if param.grad is not None:
        #         param.grad.add_(grad)
        #     else:
        #         grad = grad.detach()
        #         grad.requires_grad_(False)
        #         param.grad = grad
        vf_loss.backward()
        return torch.zeros(state.size(0), device=state.device)

    def _forward(self, xs, agent):
        logits = self._policy[agent](xs)
        a0, s0, st0 = make_samples(logits, self._temperature)
        values = self._value_mem[agent](torch.cat([xs, s0], dim=1))
        values_t = self._value_mem[agent](torch.cat([xs, st0], dim=1))
        if self.training:
            actions = a0
        else:
            actions = logits.max(dim=1)[1]
        state = torch.stack([logits, values.expand_as(logits), values_t.expand_as(logits)], dim=2)
        return xs, actions, self._eval_stochastic_are_exp(actions, logits), state

