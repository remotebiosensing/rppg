import torch
import torchvision
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import learn2learn as l2l
from tqdm import tqdm
# writer = SummaryWriter()
from pynvml.smi import nvidia_smi
import copy

nvsmi = nvidia_smi.getInstance()
def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])
class train_model:

    def __init__(self, name, models, train_dataset, test_dataset, criterion, lr, model_path, num_epochs, device):
        self.name = name
        self.model = models
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.lr = lr
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device

        train_set  = [l2l.data.MetaDataset(self.train_dataset[0]),
                      l2l.data.MetaDataset(self.train_dataset[1])]
        train_gen = [l2l.data.TaskDataset(train_set[0], num_tasks = 2,
                                          task_transforms=[l2l.data.transforms.NWays(train_set[0],2),
                                                           l2l.data.transforms.KShots(train_set[0],1),
                                                           l2l.data.transforms.LoadData(train_set[0])]),
                     l2l.data.TaskDataset(train_set[1], num_tasks=2,
                                          task_transforms=[l2l.data.transforms.NWays(train_set[1], 2),
                                                           l2l.data.transforms.KShots(train_set[1], 1),
                                                           l2l.data.transforms.LoadData(train_set[1])])]
        test_set = [l2l.data.MetaDataset(self.test_dataset[0]),
                     l2l.data.MetaDataset(self.test_dataset[1])]
        test_gen = [l2l.data.TaskDataset(test_set[0], num_tasks=2,task_transforms=None),
                     l2l.data.TaskDataset(test_set[1], num_tasks=2,task_transforms=None)]

        self.optimizers = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)

        maml = l2l.algorithms.MAML(self.model, lr=0.0001)


        for iteration in range(25): # num iter 25
            self.optimizers.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for task in range(3): # meta_batch size
                # Compute meta-training loss
                learner = maml.clone()
                train_task = [train_gen[0].sample(), train_gen[1].sample()]
                evaluation_error, evaluation_accuracy = fast_adapt(train_task,learner,self.criterion,5,5,1,device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                #meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                batch =  [train_gen[0].sample(), train_gen[1].sample()]
                evaluation_error, evaluation_accuracy = fast_adapt(train_task,learner,self.criterion,5,5,1,device)
                meta_valid_error += evaluation_error.item()
                #meta_valid_accuracy += evaluation_accuracy.item()
            print(meta_train_error)
            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / 3)
            self.optimizers.step()

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(3):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = [train_gen[0].sample(), train_gen[1].sample()]
            evaluation_error, evaluation_accuracy = fast_adapt(train_task,learner,self.criterion,5,5,1,device)
            meta_test_error += evaluation_error.item()
            #meta_test_accuracy += evaluation_accuracy.item()



def fast_adapt(task, learner, loss, adaptation_steps, shots, ways, device):

    app_data = torch.utils.data.DataLoader(_BatchedDataset(task[0]), 1, False)
    mot_data = torch.utils.data.DataLoader(_BatchedDataset(task[1]), 1, False)

    evaluation_error = 0

    for (a, t), (m, _) in zip(app_data, mot_data):
        a,m,t = a.to(device),m.to(device),t.to(device)

        for step in range(adaptation_steps):
            adaptation_error = loss(learner(a,m),t)
            learner.adapt(adaptation_error)

        predictions = learner(a,m)
        evaluation_error += loss(predictions,t)
        print(evaluation_error)
    return evaluation_error, 0.0

def compute_loss(task,device,learner,loss_function, batch = 32):
    loss = 0.0
    acc = 0.0


    app_data = torch.utils.data.DataLoader(_BatchedDataset(task[0]),batch,False)
    mot_data = torch.utils.data.DataLoader(_BatchedDataset(task[1]),batch,False)
    for (a,t),(m,_) in zip(app_data,mot_data):
        # _BatchedDataset(task[0]),batch_size=batch, shuffle=False,num_workers=0),
        # torch.utils.data.DataLoader(_BatchedDataset(task[1]),batch_size=batch,shuffle=False,num_workers=0))):

        a,m,t = a.to(device),m.to(device),t.to(device)

        output = learner(a,m)
        # print("after amt: %s" % getMemoryUsage())

        curr_loss = loss_function(output,t)
        # print("Before amy: %s" % getMemoryUsage())
        loss += curr_loss

    return loss, acc

class _BatchedDataset(torch.utils.data.Dataset):
    def __init__(self, batched):
        self.sents = [s for s in batched[0]]
        self.ys = [y for y in batched[1]]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return (self.sents[idx], self.ys[idx])