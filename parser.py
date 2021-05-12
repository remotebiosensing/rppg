import argparse
from test import test_model
from train import train
import torch
from preprocessing import DatasetDeepPhysUBFC
from torch.utils.data import DataLoader
import torchsummary
import os
from model_functional import model as M
# from metalearning import meta_train_model
import numpy as np
from model import model
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])

# from metadataset import metadataset
import learn2learn as l2l
import metatrain as mt
def select_model():

    reply = input('Select Model\n'
                  '1. DeepPhys 2. MTTS-CAN 3. MT-CAN 4. TS-CAN')
    config = config.model_config(reply)
    if reply is '1':
        args.model = 'CAN'
    elif reply is '2':
        args.model = 'MTTS-CAN'
    elif reply is '3':
        args.model = 'MT-CAN '
    elif reply is '4':
        args.model = 'TS-CAN'
    else:
        print("error")
        exit(0)

if __name__ == '__main__':
    print("Before GPU Memory: %s" % getMemoryUsage())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CAN',
                        help='[____]CAN, [MT] : Multi task learning, [TS] : TSM Module, '
                             'ex) MTTS-CAN, MTTSCAN, CAN')
    parser.add_argument('--GPU_num', type=int, default=0, help='GPU number : 0 or 1')
    parser.add_argument('--loss', type=str, default='MSE', help='MSE')
    parser.add_argument('--data', type=str, default='/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/subject',
                        help='path to DATA')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=44, help='batch size')
    parser.add_argument('--in_channels', type=int, default=3, help='in_channels')
    parser.add_argument('--out_channels', type=int, default=32, help='out_channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel_size')
    parser.add_argument('--checkpoint_dir', type=str, default='./', help='checkpoints will be saved in this directory')
    parser.add_argument('--img_size', type=int, default=36, help='size of image')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.2, help='meta learning rate')
    parser.add_argument('--preprocessing', type=bool, default=False, help='preprocessing rate')
    parser.add_argument('--check_model', type=bool, default=True,
                        help='True : check model summary False : train or test')
    parser.add_argument('--train', type=int, default=0, help=" 0:train,False, Test")

    # parser.add_argument('--pretrained_weights', type=str, help='if specified starts from checkpoint model')
    # parser.add_argument('--crop', type=bool, default=False, help='crop with blazeFace(preprocessing step)')
    # parser.add_argument('--img_augm', type=bool, default=False, help='image augmentation(flip, color jitter)')
    # parser.add_argument('--freq_augm', type=bool, default=False, help='apply frequency augmentation')
    args = parser.parse_args()
    # dataset = metadataset('./', 'train', 16, 2, 5, 5, 0, 0)
    #select_model()

    # model = M(in_channels=3,out_channels=32,kernel_size=3).to(args.device)
    #
    # [print(block) for block in model.blocks]
    # tmp = [filter(lambda x: x.requires_grad, block.parameters()) for block in model.blocks]
    # num = [sum(map(lambda x: np.prod(x.shape), t)) for t in tmp]
    #print('Total trainable tensors:', sum(num))



    reply = input('Select Train/test mode\n'
                  '1. train 2. test 3. meta-train 4. meta-test')
    if reply is '1':
        args.train = 1
    elif reply is '2':
        args.train = 2
    elif reply is '3':
        args.train = 3
    elif reply is '4':
        args.train = 4
    else:
        print("error")
        exit(0)

    if args.train is True:
        if args.checkpoint_dir:
            try:
                os.makedirs(f'checkpoints/{args.checkpoint_dir}')
                print("Output directory is created")
            except FileExistsError:
                reply = input('Override existing weights? [y/n]')
                if reply == 'n':
                    print('Add another output path then!')
                    exit(0)

    hyper_params = {
        "model": args.model,
        "loss_fn": args.loss,
        "database": args.data,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "kernel_size": args.kernel_size,
        "checkpoint_dir": args.checkpoint_dir,
        "img_size": args.img_size,
        "learning_rate": args.lr,
        "preprocessing": args.preprocessing
        # "crop": args.crop,
        # "img_augm": args.img_augm,
        # "freq_augm": args.freq_augm
    }

    device = torch.device('cuda:' + str(args.GPU_num)) if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    print(args.model)
    Dataset = None

    # --------------------------
    # Load model
    # --------------------------

    models = model(in_channels=args.in_channels, out_channels=args.out_channels, kernel_size=args.kernel_size,
                   model=args.model).to(device)
    print("Before GPU Memory: %s" % getMemoryUsage())
    if args.check_model is True:
        models.to(device)
        torchsummary.summary(models, ((3, 36, 36), (3, 36, 36)), )
        print('\ncheck model architecture')
    print("Before GPU Memory: %s" % getMemoryUsage())

    loss_fn = None
    if args.loss == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        print('\nError! No such loss function. Choose from : L1, MSE')
        exit(666)

    print('Constructing data loader for DeepPhys architecture....')



    Dataset = DatasetDeepPhysUBFC(args.data,
                                  img_size=args.img_size,
                                  preprocessing=args.preprocessing,
                                  train=args.train)
    Dataset = Dataset()



    if args.train is 1:# train
        train_loader = DataLoader(Dataset[0], batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(Dataset[1], batch_size=args.batch_size, shuffle=False)

        print('\nDataLoaders successfully constructed!')

        train(args.model,models, train_loader, val_loader, loss_fn, args.lr, args.checkpoint_dir,args.epochs,device=device)
    elif args.train is 2:# test
        test_loader = DataLoader(Dataset, batch_size=1, shuffle=False)

        print('\nDataLoaders successfully constructed!')

        checkpoint = torch.load("checkpoint_train.pth")
        models.load_state_dict(checkpoint['state_dict'])
        test_model(args.model,models, test_loader, loss_fn,  args.checkpoint_dir,
                   args.epochs, device)
    elif args.train is 3:# meta train
        maml = l2l.algorithms.MAML(models, lr = 0.1)
        #train_loader = DataLoader(Dataset[0], batch_size=args.batch_size, shuffle=False)
        #test_loader = DataLoader(Dataset[1], batch_size=args.batch_size, shuffle=False)
        mt.train_model(args.model, models,Dataset[0:2],Dataset[2:4],loss_fn,args.lr,args.meta_lr,args.epochs,device)
    elif args.train is 4:# meta test
        train_loader = DataLoader(Dataset[0], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(Dataset[1], batch_size=args.batch_size, shuffle=False)