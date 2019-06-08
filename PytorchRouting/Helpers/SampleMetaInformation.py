"""
This file defines class SampleMetaInformation.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/6/18
"""
from collections import defaultdict


class SampleMetaInformation(object):
    """
    Class SampleMetaInformation should be used to store metainformation for each sample.
    """

    def __init__(self, task=None):
        self.task = task
        self.steps = []

    def append(self, attr_name, obj, new_step=False):
        if new_step:
            self.steps.append({})
        else:
            assert len(self.steps) > 0, 'initialize a new step first by calling this function with new_step=True'
        self.steps[-1][attr_name] = obj

    def finalize(self):
        """
        This method finalizes a trajectory, by translating the stored sar tuples into attributes of this class
        :return:
        """
        res = {}
        for step in self.steps:
            for key in step.keys():
                res[key] = []
        for i, step in enumerate(self.steps):
            for key in res.keys():
                if key not in step:
                    res[key].append(None)
                else:
                    res[key].append(step[key])
        for key, val in res.items():
            setattr(self, key, val)
