import torch
from rppg.optim import optimizer
from rppg.loss import loss_fn
from copy import deepcopy

class MAML:
    def __init__(self, model, inner_optim, inner_loss, inner_lr=0.01, num_updates=5):
        self.model = model
        self.inner_optim = inner_optim
        self.inner_loss = inner_loss
        self.inner_lr = inner_lr
        self.num_updates = num_updates

    def inner_update(self, task):
        task_model = deepcopy(self.model)
        task_optim = optimizer(task_model.parameters(), learning_rate=self.inner_lr,optim=self.inner_optim)
        task_loss = loss_fn(self.inner_loss)
        for _ in range(self.num_updates):
            for x_batch, y_batch in task:
                task_optim.zero_grad()
                loss = task_loss(task_model(x_batch), y_batch)
                loss.backward()
                task_optim.step()

        return task_model

    def meta_update(self, tasks):
        meta_grads = []
        outter_loss = loss_fn(self.inner_loss)
        outter_optim = optimizer(self.model.parameters(), learning_rate=self.inner_lr,optim=self.inner_optim)
        for task in tasks:
            loss = 0.0
            task_model = self.inner_update(task)
            for x_batch, y_batch in task:
                loss += outter_loss(task_model(x_batch), y_batch)
            loss /= len(task)
            grads = torch.autograd.grad(loss, task_model.parameters(), retain_graph=True)
            meta_grads.append(grads)
        outter_optim.zero_grad()
        for (name, param), grads in zip(self.model.named_parameters(), zip(*meta_grads)):
            grad = torch.stack([g for g in grads]).mean(dim=0)
            if param.grad is None:
                param.grad = grad.detach()
            else:
                param.grad += grad.detach()

        outter_optim.step()
