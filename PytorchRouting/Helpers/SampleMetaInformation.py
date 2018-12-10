"""
This file defines class SampleMetaInformation.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/6/18
"""


class SampleMetaInformation(object):
    """
    Class SampleMetaInformation should be used to store metainformation for each sample.
    """

    def __init__(self, task=None):
        self.task = task
        self.add_rewards = []

    def append(self, attr_name, obj):
        try:
            getattr(self, attr_name).append(obj)
        except AttributeError:
            setattr(self, attr_name, [obj])
