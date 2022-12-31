import torch
from vid2bp.LSTMAutoEncoder.dataset.LSTMAutoEncoder_dataset_loader import dataset_loader
from vid2bp.nets.modules.LSTMAutoEncoder import LSTMAutoEncoder
from vid2bp.LSTMAutoEncoder.train_fn import train_fn, test_fn
from loss import NegPearsonLoss
import matplotlib.pyplot as plt
import numpy as np
import wandb


def main(label='abp', batch_size=4096, learning_rate=0.0025, epochs=50, loss='MSE', wandb_flag=True):
    '''
    set hyperparameter
    '''
    learning_rate = learning_rate
    batch_size = batch_size
    epochs = epochs

    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')

    print('predict label: ', label)

    '''
    load model
    '''
    model = LSTMAutoEncoder(hidden_size=128, length=360, num_layers=1, label=label).cuda()
    if label == 'abp':
        model.load_state_dict(torch.load('/home/najy/rppg/model_weights/LSTMAutoEncoder_best.pth'))
        print('load ppg2ppg  model weights')
    '''
    load dataset
    '''
    data_loaders = dataset_loader(channel=1, batch_size=batch_size, label=label)
    '''
    set training parameters
    '''
    if loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss == 'neg_pearson':
        criterion = [NegPearsonLoss(), torch.nn.MSELoss()]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    min_val_loss = 100.0
    min_test_loss = 100.0

    '''
    set wandb parameters
    '''
    wandb.init(project='torch_rPPG', entity="najubae777",
               name="LSTMAutoEncoder_predict_" + label + "_MSE+" + loss)
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": loss
    }

    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    test_epochs = []
    for epoch in range(epochs):
        train_loss = train_fn(epoch, model, optimizer, criterion, data_loaders[0], "Train", wandb_flag=wandb_flag)
        train_loss_list.append(train_loss)
        val_loss = test_fn(epoch, model, criterion, data_loaders[1], "Val", wandb_flag=wandb_flag)
        val_loss_list.append(val_loss)

        plt.subplot(2, 1, 1)
        plt.title("epoch " + str(epoch + 1) + " train loss")
        plt.plot(np.arange(epoch + 1), train_loss_list, label="train")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title("epoch " + str(epoch + 1) + " valid loss")
        plt.plot(np.arange(epoch + 1), val_loss_list, label="val")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if min_val_loss > val_loss and epoch > 10:
            min_val_loss = val_loss
            test_epochs.append(epoch)
            running_loss = test_fn(epoch, model, criterion, data_loaders[2], "Test")
            test_loss_list.append(running_loss)
            plt.title("epoch " + str(epoch + 1) + " test loss")
            plt.plot(test_epochs, test_loss_list, label="test")
            plt.legend()
            plt.show()
            if min_test_loss > running_loss:
                min_test_loss = running_loss
                torch.save(model.state_dict(),
                           "/home/najy/rppg/model_weights/LSTMAutoEncoder_predict_" + label + "_best.pth")
        if epoch % 10 == 0 or epoch == epochs - 1:
            running_loss = test_fn(epoch, model, criterion, data_loaders[2], "Test")
            test_epochs.append(epoch)
            test_loss_list.append(running_loss)
            plt.title("epoch " + str(epoch + 1) + " test loss")
            plt.plot(test_epochs, test_loss_list, label="test")
            plt.legend()
            plt.show()
            if min_test_loss > running_loss:
                min_test_loss = running_loss
                torch.save(model.state_dict(),
                           "/home/najy/rppg/model_weights/LSTMAutoEncoder_predict_" + label + "_best.pth")

    wandb.finish()


if __name__ == '__main__':
    main(label='abp', batch_size=4096, learning_rate=0.0025, epochs=50, loss='neg_pearson', wandb_flag=True)
