"""
This file defines some simple experiments to illustrate how Pytorch-Routing functions.
"""
import numpy as np
import torch
from PytorchRouting.Examples.Models import PerTask_all_fc, WPL_routed_all_fc
from PytorchRouting.Examples.Datasets import MNIST_MTL, CIFAR100MTL


def compute_batch(model, batch):
    samples, labels, tasks = batch
    out, meta = model(samples, tasks=tasks)
    correct_predictions = (out.max(dim=1)[1].squeeze() == labels.squeeze()).cpu().numpy()
    accuracy = correct_predictions.sum()
    module_loss, decision_loss = model.loss(out, labels, meta)
    return module_loss, decision_loss, accuracy


def run_experiment(model, dataset):
    learning_rates = {0: 1e-2, 10: 1e-3, 20: 3e-4, 40: 1e-4}
    print('Loaded dataset and constructed model. Starting Training ...')
    for epoch in range(50):
        if epoch in learning_rates:
            opt = torch.optim.SGD(model.parameters(), lr=learning_rates[epoch])
        train_log, test_log = np.zeros((3,)), np.zeros((3,))
        train_samples_seen, test_samples_seen = 0, 0
        dataset.enter_train_mode()
        while True:
            try:
                batch = dataset.get_batch()
            except StopIteration:
                break
            train_samples_seen += len(batch[0])
            module_loss, decision_loss, accuracy = compute_batch(model, batch)
            train_log += np.array([module_loss.tolist(), decision_loss.tolist(), accuracy])
            (module_loss + decision_loss).backward()
            opt.step()
            model.zero_grad()
        dataset.enter_test_mode()
        while True:
            try:
                batch = dataset.get_batch()
            except StopIteration:
                break
            test_samples_seen += len(batch[0])
            module_loss, decision_loss, accuracy = compute_batch(model, batch)
            test_log += np.array([module_loss.tolist(), decision_loss.tolist(), accuracy])
        print('Epoch {} finished.\n'
              '    Training averages: Model loss: {}, Routing loss: {}, Accuracy: {}\n'
              '    Testing averages:  Model loss: {}, Routing loss: {}, Accuracy: {}'.format(
            epoch + 1, *(train_log/train_samples_seen).round(3), *(test_log/test_samples_seen).round(3)))


if __name__ == '__main__':
    # MNIST
    # dataset = MNIST_MTL(64, data_files=['./Datasets/mnist.pkl.gz'])
    # model = PerTask_all_fc(1, 288, 2, dataset.num_tasks, dataset.num_tasks)
    # model = WPL_routed_all_fc(1, 288, 2, dataset.num_tasks, dataset.num_tasks)

    # CIFAR
    dataset = CIFAR100MTL(64, data_files=['./Datasets/cifar-100-py/train', './Datasets/cifar-100-py/test'])
    # model = PerTask_all_fc(3, 512, 5, dataset.num_tasks, dataset.num_tasks)
    model = WPL_routed_all_fc(3, 512, 5, dataset.num_tasks, dataset.num_tasks)

    model.cuda()
    run_experiment(model, dataset)
