"""
This file defines class Sample.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/8/18
"""


class Sample(object):
    """
    Class Sample defines ...
    """

    def __init__(
            self,
            loss_function,
            state,
            action,
            reward,
            cum_return,
            next_state,
            next_action
        ):
        self.loss_function = loss_function
        self.state = state
        self.action = action
        self.reward = reward
        self.cum_return = cum_return
        self.next_state = next_state
        self.next_action = next_action
