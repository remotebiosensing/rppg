"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils.funcs import normalize,plot_graph

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, i , mode):

    batch['train'][0] = batch['train'][0].squeeze()
    batch['test'][0] = batch['test'][0].squeeze()
    batch['train'][1] = batch['train'][1].squeeze()
    batch['test'][1] = batch['test'][1].squeeze()

    inference_array = []
    target_array = []

    #data, labels = batch
    #data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    #adaptation_indices = np.zeros(data.size(0), dtype=bool)
    #adaptation_indices[np.arange(shots*ways) * 2] = True
    #evaluation_indices = torch.from_numpy(~adaptation_indices)
    #adaptation_indices = torch.from_numpy(adaptation_indices)
    #adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    #evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    adaptation_data, adaptation_labels = batch['train']
    evaluation_data, evaluation_labels = batch['test']

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    #valid_accuracy = accuracy(predictions, evaluation_labels)



    if mode =='test'and i % 30 == 1:
        plt.rcParams["figure.figsize"] = (14, 5)
        #plt.plot(peaks, np.array(inference_array)[peaks], 'xr');
        plt.plot(normalize(predictions.cpu().detach().numpy()[0]))
        #plt.plot(peaks2, np.array(target_array)[peaks2], 'xb');
        plt.plot(normalize(evaluation_labels.cpu().detach().numpy()[0]))
        plt.show()
        #print(len(peaks), len(peaks2))

    return valid_error


def Meta(
        model,
        train_loader,
        test_loader,
        inner_criterion,
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    # Create model
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = inner_criterion #nn.CrossEntropyLoss(reduction='mean')

    i = 0
    with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
    #for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0

        #for task in range(meta_batch_size):
        for batch in tepoch:

            # Compute meta-training loss
            learner = maml.clone()
            i = 0 #i += 1
            evaluation_error = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               i,'train')
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            #meta_train_accuracy += evaluation_accuracy.item()
            tepoch.set_postfix(loss=meta_train_error)

        #print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / len(tepoch))
        #print('Meta Train Accuracy', meta_train_accuracy / len(tepoch))
        #print('Meta Valid Error', meta_valid_error / meta_batch_size)
        #rint('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / len(tepoch))
        opt.step()

    meta_test_error = 0.0


    with tqdm(test_loader, desc="Test ", total=len(test_loader)) as tepoch:
        for batch in tepoch:
        #for task in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            evaluation_error = fast_adapt(batch,
                                           learner,
                                           loss,
                                           adaptation_steps,
                                           i,'test')
            meta_test_error += evaluation_error.item()
            i += 1
        print('Meta Test Error', meta_test_error / len(tepoch))
        print('\n')
