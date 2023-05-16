import torch
from rppg.optim import optimizer
from rppg.loss import loss_fn
from copy import deepcopy
from tqdm import tqdm


class MAML:
    def __init__(self, model, inner_optim, outer_optim, inner_loss, inner_lr=0.005, outer_lr=0.001, num_updates=10):
        self.model = model  # e.g. pretrained model
        self.inner_optim = inner_optim
        self.outer_optim = outer_optim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_loss = inner_loss
        self.num_updates = num_updates

    def inner_update(self, support_tasks):
        individual_model = deepcopy(self.model)
        task_optim = optimizer(individual_model.parameters(), learning_rate=self.inner_lr, optim=self.inner_optim)
        task_loss = loss_fn(self.inner_loss)
        for _ in range(self.num_updates):
            for x_batch, y_batch in support_tasks:
                task_optim.zero_grad()
                loss = task_loss(individual_model(x_batch), y_batch)
                loss.backward()
                task_optim.step()

        return individual_model

    def meta_update(self, support_tasks, query_tasks):
        for _ in tqdm(range(self.num_updates)):
            meta_grads = []
            outer_loss = loss_fn(self.inner_loss)
            outer_optim = optimizer(self.model.parameters(), learning_rate=self.outer_lr, optim=self.outer_optim)

            for q in query_tasks:
                loss = 0.0
                individual_model = self.inner_update(support_tasks)
                for x_batch, y_batch in q:
                    loss += outer_loss(individual_model(x_batch), y_batch)
                loss /= len(q)
                grads = torch.autograd.grad(outputs=loss, inputs=individual_model.parameters(), retain_graph=True)
                meta_grads.append(grads)
            outer_optim.zero_grad()

            for (name, param), grads in zip(self.model.named_parameters(), zip(*meta_grads)):
                grad = torch.stack([g for g in grads]).mean(dim=0)
                if param.grad is None:
                    param.grad = grad.detach()
                else:
                    param.grad += self.outer_lr * grad.detach()

            outer_optim.step()

        # return self.model
