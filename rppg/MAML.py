import torch
from rppg.optim import optimizer
from rppg.loss import loss_fn
from copy import deepcopy
from tqdm import tqdm

class MAML:
    def __init__(self, model, inner_optim, outer_optim, inner_loss, outer_loss, inner_lr=0.005, outer_lr=0.001, num_updates=5):
        self.model = model  # e.g. pretrained model
        self.inner_optim = inner_optim
        self.outer_optim = outer_optim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_loss = inner_loss
        self.outer_loss = outer_loss
        self.num_updates = num_updates

    def inner_update(self, support_task):
        individual_model = deepcopy(self.model)
        task_optim = optimizer(individual_model.parameters(), learning_rate=self.inner_lr, optim=self.inner_optim)
        task_loss = loss_fn(self.inner_loss)
        for _ in range(self.num_updates):
            for x_batch, y_batch in support_task:
                task_optim.zero_grad()
                loss = task_loss(individual_model(x_batch), y_batch)
                loss.backward()
                task_optim.step()

        return individual_model

    def meta_update(self, tasks, epoch):
        meta_grads = []
        outter_loss = loss_fn(self.outer_loss)
        outter_optim = optimizer(self.model.parameters(), learning_rate=self.outer_lr, optim=self.outer_optim)
        with tqdm(total=len(tasks), position=0, leave=True, desc=f"epoch {epoch}") as pbar:
            for support_task, query_task in tasks:
                loss = 0.0
                individual_model = self.inner_update(support_task)

                for x_batch, y_batch in query_task:
                    loss += outter_loss(individual_model(x_batch), y_batch)
                loss /= len(tasks)
                grads = torch.autograd.grad(loss, individual_model.parameters(), retain_graph=True)
                meta_grads.append(grads)
                pbar.update(1)

            outter_optim.zero_grad()
            for (name, param), grads in zip(self.model.named_parameters(), zip(*meta_grads)):
                grad = torch.stack([g for g in grads]).mean(dim=0)
                if param.grad is None:
                    param.grad = grad.detach()
                else:
                    # param.grad += grad.detach()
                    param.grad += self.outer_lr * grad.detach()

            outter_optim.step()


