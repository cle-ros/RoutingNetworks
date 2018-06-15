"""
This file defines the base class `Dataset` and classes for the MNIST and CIFAR100 MTL versions.
As this is mostly for demonstration purposes, the code is uncommented.

@author: Clemens Rosenbaum :: cgbr@cs.umass.edu
@created: 6/14/18
"""
import abc
import gzip
import random
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

import torch
from torch.autograd import Variable


class Dataset(object, metaclass=abc.ABCMeta):
    """
    Class Datasets defines ...
    """

    def __init__(self, batch_size, data_files=()):
        self._iterator = None
        self._batch_size = batch_size
        self._data_files = data_files
        self._train_set, self._test_set = self._get_datasets()

    @abc.abstractmethod
    def _get_datasets(self): return [], []

    @staticmethod
    def _batched_iter(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            samples = Variable(torch.stack([torch.FloatTensor(sample[0]) for sample in batch], 0)).cuda()
            targets = Variable(torch.stack([torch.LongTensor([sample[1]]) for sample in batch], 0)).cuda()
            tasks = [sample[2] for sample in batch]
            yield samples, targets, tasks

    def get_batch(self):
        return next(self._iterator)

    def enter_train_mode(self):
        self._iterator = self._batched_iter(self._train_set, self._batch_size)

    def enter_test_mode(self):
        self._iterator = self._batched_iter(self._test_set, self._batch_size)


class MNIST_MTL(Dataset):
    def __init__(self, *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)
        self.num_tasks = 10

    @staticmethod
    def _process_list_of_samples(samples):
        processed = []
        for s in samples:
            s = np.array(s).reshape((1, 28, 28))
            processed.append(s)
        return processed

    def _get_datasets(self):
        with gzip.open(self._data_files[0], 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            (train_samples, train_labels), _, (test_samples, test_labels) = u.load()
        train_samples = self._process_list_of_samples(train_samples)
        test_samples = self._process_list_of_samples(test_samples)
        mtl_train_set = []
        mtl_test_set = []
        for task in range(10):
            for sample, ground_label in zip(train_samples, train_labels):
                label = int(ground_label == task)
                mtl_train_set.append((sample, label, task))
            for sample, ground_label in zip(test_samples, test_labels):
                label = int(ground_label == task)
                mtl_test_set.append((sample, label, task))
        random.shuffle(mtl_train_set)
        random.shuffle(mtl_test_set)
        return mtl_train_set, mtl_test_set


class CIFAR100MTL(Dataset):
    def __init__(self, *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)
        self.num_tasks = 20

    def _get_datasets(self):
        datasets = []
        for fn in self._data_files:  # assuming that the datafiles are [train_file_name, test_file_name]
            samples, labels, tasks = [], [], []
            with open(fn, 'rb') as f:
                data_dict = pickle.load(f, encoding='latin1')
            samples += [np.resize(s, (3, 32, 32)) for s in data_dict['data']]
            tasks += [int(fl) for fl in data_dict['coarse_labels']]
            labels += [int(cl) % 5 for cl in data_dict['fine_labels']]
            datasets.append(list(zip(samples, labels, tasks)))
        train_set, test_set = datasets
        return train_set, test_set
