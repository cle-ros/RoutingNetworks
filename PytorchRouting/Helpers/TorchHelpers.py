import torch.nn as nn


class FakeFlatModuleList(nn.ModuleList):
    """
    A little util class inherited from nn.ModuleList that returns the same object for any index requested.
    """
    def __getitem__(self, idx):
        assert len(self._modules) == 1, 'Fake ModuleList with more than one module instantiated. Aborting.'
        if isinstance(idx, slice):
            raise ValueError('cannot slice into a FakeModuleList')
        else:
            return self._modules['0']


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)

    def forward(self, xs):
        return xs
