import torch
from vid2bp.PPG2ABP.dataset.PPG2ABP_dataset_loader import dataset_loader
from vid2bp.nets.modules import MultiResUNet1D
from vid2bp.PPG2ABP.utils.ref_train_fn import train_fn, test_fn
import matplotlib.pyplot as plt
import numpy as np
from vid2bp.nets.modules import UNetDS64


def main():
    learning_rate = 0.001
    batch_size = 1024
    epochs = 50
    length = 352

    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')

    '''
    load model
    '''
    model = MultiResUNet1D.MultiResUNet1D().cuda()
    app_model = UNetDS64.UNetDS64(352)
    if torch.cuda.is_available():
        app_model = app_model.cuda()
    app_model.load_state_dict(torch.load('/home/najy/PycharmProjects/PPG2ABP_weights/UNetDS64_0.31550129055976867.pth'))
    '''
    load dataset
    '''
    data_loaders = dataset_loader(channel=1, batch_size=batch_size, dataset_name='MultiResUNet1D')

    '''
    set training parameters
    '''
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    min_val_loss = 10000.0
    min_test_loss = 10000.0
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    test_epochs = []
    for epoch in range(epochs):
        train_loss = train_fn(epoch, app_model, model, optimizer, criterion, data_loaders[0], "Train")
        train_loss_list.append(train_loss)
        val_loss = test_fn(epoch, app_model, model, criterion, data_loaders[1], "Val")
        val_loss_list.append(val_loss)

        plt.subplot(2, 1, 1)
        plt.title("epoch " + str(epoch) + " train loss")
        plt.plot(np.arange(epoch + 1), train_loss_list, label="train")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title("epoch " + str(epoch) + " valid loss")
        plt.plot(np.arange(epoch + 1), val_loss_list, label="val")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if min_val_loss > val_loss and epoch > 5:
            min_val_loss = val_loss
            test_epochs.append(epoch)
            running_loss = test_fn(epoch, app_model, model, criterion, data_loaders[2], "Test")
            test_loss_list.append(running_loss)
            plt.title("epoch " + str(epoch) + " test loss")
            plt.plot(test_epochs, test_loss_list, label="test")
            plt.legend()
            plt.show()
            if min_test_loss > running_loss:
                min_test_loss = running_loss
                torch.save(model.state_dict(),
                           "/home/najy/PycharmProjects/PPG2ABP_weights/MultiResUNet1D_" + str(running_loss) + ".pth")

        if epoch % 10 == 0 or epoch == epochs - 1:
            running_loss = test_fn(epoch, app_model, model, criterion, data_loaders[2], "Test")
            test_epochs.append(epoch)
            test_loss_list.append(running_loss)
            plt.title("epoch " + str(epoch) + " test loss")
            plt.plot(test_epochs, test_loss_list, label="test")
            plt.legend()
            plt.show()
            if min_test_loss > running_loss:
                min_test_loss = running_loss
                torch.save(model.state_dict(),
                           "/home/najy/PycharmProjects/PPG2ABP_weights/MultiResUNet1D_" + str(running_loss) + ".pth")


if __name__ == '__main__':
    main()
