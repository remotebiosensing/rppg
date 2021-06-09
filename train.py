import torch
import torchvision
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, val_loader, criterion, optimizer, model_path, epochs, device):
    writer = SummaryWriter()
    folder_name = datetime.datetime.now()
    model = model.to(device)
    tmp_val_loss = 100
    for epoch in range(epochs):
        running_loss = 0.0
        for i_batch, (avg, mot, lab) in tqdm(enumerate(train_loader), desc="Train ", total=len(train_loader)):
            optimizer.zero_grad()
            avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
            # if i_batch is 0 and epoch is 0:
            #     avg_grid = torchvision.utils.make_grid(avg[:5], nrow=1)
            #     writer.add_image('Appearance', avg_grid)
            #     mot_grid = torchvision.utils.make_grid(mot[:5], nrow=1)
            #     writer.add_image('Motion', mot_grid)
            output = model(avg, mot)
            # if i_batch is 0:
            #     mask1, mask2 = model.appearance_model(avg)
            #     writer.add_image('mask1', mask1[0], epoch)
            #     writer.add_image('mask2', mask2[0], epoch)
            loss = criterion(output[0], lab)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        print('Train : [%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, running_loss / 32))
        writer.add_scalar('train_loss', running_loss, epoch)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for j, (val_A, val_M, val_T) in tqdm(enumerate(val_loader), desc="Validation ", total=len(val_loader)):
                val_A, val_M, val_T = val_A.to(device), val_M.to(device), val_T.to(device)
                # if j is 0 and epoch is 0:
                    # val_A_grid = torchvision.utils.make_grid(val_A[:3], nrow=3)
                    # writer.add_image('val_A_grid', val_A_grid)
                    # val_M_grid = torchvision.utils.make_grid(val_M[:3], nrow=3)
                    # writer.add_image('val_M_grid', val_M_grid)
                val_output = model(val_A, val_M)
                if j is 0:
                    output_mask1_grid = torchvision.utils.make_grid(val_output[1][:2], nrow=2)
                    writer.add_image('output_mask1_grid', output_mask1_grid)
                    # output_mask2_grid = torchvision.utils.make_grid(val_output[2][:3], nrow=3)
                    # writer.add_image('output_mask2_grid', output_mask2_grid)
                    val_mask1, val_mask2 = model.appearance_model(val_A)
                    val_mask1_grid = torchvision.utils.make_grid(val_mask1[:2], nrow=2)
                    writer.add_image('val_mask1', val_mask1_grid)
                    # val_mask2_grid = torchvision.utils.make_grid(val_mask2[:3], nrow=3)
                    # writer.add_image('val_mask2', val_mask2_grid)
                v_loss = criterion(val_output[0], val_T)
                val_loss += v_loss.item()
            print('Validation : [%d, %5d] loss: %.3f' % (epoch + 1, j + 1, val_loss / 32))
            writer.add_scalar('validation_loss', val_loss, epoch)
            if tmp_val_loss > (val_loss / 32):
                checkpoint = {'Epoch': epoch,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint,
                           model_path + "/checkpoint_" + str(folder_name.day) + "d_"
                           + str(folder_name.hour) + "h_"
                           + str(folder_name.minute) + 'm.pth')
                tmp_val_loss = val_loss / 32
                print("Update tmp : " + str(tmp_val_loss))
        model.train()
    writer.close()
    print('Finished Training')
