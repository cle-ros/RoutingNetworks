"""
This file defines class RLSample.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""


class RLSample(object):
    """
    RLSample defines a simple struct-like class that is used to combine RL-relevant training information.
    (i.e. state, action, reward, next state, next action)
    """

    def __init__(
            self,
            loss_function,
            state,
            action,
            reward,
            cum_return,
            prior_action,
            next_state,
            next_action
        ):
        self.loss_function = loss_function
        self.state = state
        self.action = action
        self.prior_action = prior_action
        self.reward = reward
        self.cum_return = cum_return
        self.next_state = next_state
        self.next_action = next_action
