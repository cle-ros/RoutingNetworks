"""
This file defines class Meta.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/6/18
"""


class Meta(object):
    """
    Class Meta should be used to attach metainformation to samples
    """

    def __init__(self):
        pass

    def append(self, attr_name, obj):
        try:
            getattr(self, attr_name).append(obj)
        except AttributeError:
            setattr(self, attr_name, [obj])
