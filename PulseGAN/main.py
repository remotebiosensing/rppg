import argparse
import os

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

#from data_preprocess import sample_rate
from GAN_model import Generator, Discriminator
from utils import AudioDataset, emphasis
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=4, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=80, type=int, help='train epochs number')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    #g_optimizer = optim.RMSprop(generator.parameters(), lr=0.001)
    #d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.001)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    g_scheduler = ReduceLROnPlateau(g_optimizer, factor =0.1, patience = 3)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    d_scheduler = ReduceLROnPlateau(d_optimizer, factor =0.1, patience = 3)

    for epoch in range(NUM_EPOCHS):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:

            # latent vector - normal distribution
            z = nn.init.normal(torch.Tensor(train_batch.size(0), 64, 4))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            #z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            clean_loss.backward()
            # TRAIN D to recognize generated audio as noisy
            #generated_outputs = generator(train_noisy, z)
            generated_outputs = generator(train_noisy)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            noisy_loss.backward()

            d_loss = 0.5*(clean_loss + noisy_loss)
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy)
            #generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 10 * torch.mean(l1_dist)  # conditional loss
            # spectrum loss
            l1_dist2 = torch.abs(torch.add(torch.rfft(generated_outputs,signal_ndim =1), torch.neg(torch.rfft(train_clean,signal_ndim =1))))
            g_cond_loss2 = 5 * torch.mean(l1_dist2)
            g_loss = g_loss_ + g_cond_loss + g_cond_loss2

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()

            train_bar.set_description(
                'Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
                    .format(epoch + 1, clean_loss.data, noisy_loss.data, g_loss.data, g_cond_loss.data))
            d_scheduler.step(d_loss.item())
        # TEST model
        test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        for test_file_names, test_noisy in test_bar:
            z = nn.init.normal(torch.Tensor(test_noisy.size(0), 64, 4))
            if torch.cuda.is_available():
                test_noisy, z = test_noisy.cuda(), z.cuda()
            test_noisy, z = Variable(test_noisy), Variable(z)
            #fake_rppg = generator(test_noisy, z).data.cpu().numpy()  # convert to numpy array
            fake_rppg = generator(test_noisy).data.cpu().numpy()  # convert to numpy array
            
            for idx in range(fake_rppg.shape[0]):
                generated_sample = fake_rppg[idx]
                with open('./results/generate_{}_{}.csv'.format(test_file_names[idx].replace('.npy', ''), epoch + 1),'w') as f:
                    wr = csv.writer(f)
                    wr.writerow(generated_sample.T)

        # save the model parameters for each epoch
        g_path = os.path.join('epochs', 'generator-{}.pkl'.format(epoch + 1))
        d_path = os.path.join('epochs', 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
