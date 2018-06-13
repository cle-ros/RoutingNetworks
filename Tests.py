import gzip
from collections import deque
try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.misc import imresize
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from PytorchRouting.UtilLayers import Sequential

from PytorchRouting.CoreLayers import InitializationLayer, RoutingLossLayer, SelectionLayer
from PytorchRouting.DecisionLayers import REINFORCE, QLearning, SARSA, ActorCritic, GumbelSoftmax, PerTaskAssignment, \
    WPL
from PytorchRouting.RewardFunctions.Final import NegLossReward
from PytorchRouting.RewardFunctions.PerAction import CollaborationReward


class MNIST(object):
    def _process_list_of_samples(self, samples):
        processed = []
        for s in samples:
            s = np.array(s).reshape((1, 28, 28))
            processed.append(s[None])
        return processed

    def load_data(self, file_name=None):
        with gzip.open(file_name, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            (train_samples, train_labels), (valid_samples, valid_labels), (test_samples, test_labels) = u.load()
        train_samples = self._process_list_of_samples(train_samples)
        test_samples = self._process_list_of_samples(test_samples)
        valid_samples = self._process_list_of_samples(valid_samples)
        return (train_samples, train_labels), (valid_samples, valid_labels), (test_samples, test_labels)


class RaviConvNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, flatten=False):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self._flatten = flatten

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.maxpool(y)
        if self._flatten:
            y = y.view(y.size()[0], -1)
        return y


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.convolutions = nn.Sequential(
            RaviConvNetBlock(1, 32, 3),
            RaviConvNetBlock(32, 32, 3),
            RaviConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
        )

        routing_input_dim = 288
        first_layer_width = 3
        out_dim = 10

        self.routing_layers = Sequential(
            PerTaskAssignment(),
            ActorCritic(first_layer_width, routing_input_dim, num_agents=1, exploration=0.1,
                policy_storage_type='approx', detach=False, approx_hidden_dims=(128, 128),),
            SelectionLayer([(nn.Linear, (routing_input_dim, 128)) for _ in range(first_layer_width)]),
            SelectionLayer([(nn.Linear, (128, 128)) for _ in range(first_layer_width)]),
            SelectionLayer([(nn.Linear, (128, out_dim)) for _ in range(first_layer_width)]),
        )

        self._loss_layer = RoutingLossLayer(torch.nn.CrossEntropyLoss, NegLossReward, {}, {}, 1.)

    def forward(self, x):
        y = self.convolutions(x)
        y = y.view(y.size()[0], -1)
        y, meta = self.routing_layers(y, tasks=[0 for _ in y])
        return y, meta

    def loss(self, yhat, ytrue, ym):
        return self._loss_layer(yhat, ytrue, ym)


if __name__ == '__main__':
    data = MNIST().load_data('./mnist.pkl.gz')
    model = Model()
    model.cuda()
    batch_size = 64
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    print('Loaded dataset and constructed model. Starting Training ...')
    for epoch in range(20):
        batch_loss = 0.
        log_losses = np.zeros((2,))
        for i, (sample, label) in enumerate(zip(data[0][0], data[0][1])):
            out, meta = model(Variable(torch.FloatTensor(sample).cuda()))
            label = torch.LongTensor([label]).cuda()
            loss, rloss = model.loss(out, label, meta)
            log_losses += np.array([loss.tolist()[0], rloss.tolist()[0]])
            batch_loss = batch_loss + loss + rloss
            if i % batch_size == 0 or i == len(data[0][0]) - 1:
                batch_loss.backward()
                opt.step()
                batch_loss = 0.
                model.zero_grad()
        log_losses /= float(i)
        log_losses = np.round(log_losses, 3)
        print('Epoch {} finished. Model loss: {}, Routing loss: {}'.format(epoch + 1, *log_losses))
        losses, rlosses = 0., 0.
