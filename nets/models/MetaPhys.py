import torch

from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
from nets.models.sub_models.LinearModel import LinearModel
from nets.models.sub_models.MotionModel import MotionModel_TS

from utils.funcs import normalize, plot_graph, detrend
import numpy as np
import higher

class TSCAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.attention_mask1 = None
        self.attention_mask2 = None

        self.appearance_model = AppearanceModel_2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=self.kernel_size)
        self.motion_model = MotionModel_TS(in_channels=self.in_channels, out_channels=self.out_channels,
                                           kernel_size=self.kernel_size)

        self.linear_model = LinearModel(in_channel=3136)

    def forward(self, inputs):
        """
        :param inputs:
        inputs[0] : appearance_input
        inputs[1] : motion_input
        :return:
        original 2d model
        """
        inputs = torch.chunk(inputs,2,dim=1)
        self.attention_mask1, self.attention_mask2 = self.appearance_model(torch.squeeze(inputs[0],1))
        motion_output = self.motion_model(torch.squeeze(inputs[1],1), self.attention_mask1, self.attention_mask2)
        out = self.linear_model(motion_output)

        return out

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2

def maml_train(tepoch, model, inner_criterion, outer_criterion, inner_optimizer,optimizer,  num_adapt_steps):
    model.train()

    for batch in tepoch:
        tepoch.set_description(f"Train Epoch ")

        batch['train'][0] = batch['train'][0].view(1, -1, 6, 36, 36)
        batch['test'][0] = batch['test'][0].view(1, -1, 6, 36, 36)
        batch['train'][1] = batch['train'][1].view(1, -1, 1)
        batch['test'][1] = batch['test'][1].view(1, -1, 1)

        inputs, targets = batch['train']
        test_inputs, test_targets = batch['test']

        test_losses = []
        optimizer.zero_grad()
        for task_idx, (input, target, test_input, test_target) in enumerate(
                zip(inputs, targets, test_inputs, test_targets)):
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                for inner_step in range(num_adapt_steps):
                    inner_loss = inner_criterion(fmodel(input), target)
                    diffopt.step(inner_loss)
                test_logit = fmodel(test_input)
                test_loss = outer_criterion(test_logit, test_target)
                test_losses.append(test_loss.detach())
                test_loss.backward()

        optimizer.step()
        losses = sum(test_losses) / len(tepoch)
        tepoch.set_postfix(loss=losses)

def maml_val(tepoch, model, inner_criterion, outer_criterion, inner_optimizer, num_adapt_steps):
    model.train()
    test_losses = []
    for batch in tepoch:
        tepoch.set_description(f"Validation")

        batch['train'][0] = batch['train'][0].view(1, -1, 6, 36, 36)
        batch['test'][0] = batch['test'][0].view(1, -1, 6, 36, 36)
        batch['train'][1] = batch['train'][1].view(1, -1, 1)
        batch['test'][1] = batch['test'][1].view(1, -1, 1)

        inputs, targets = batch['train']
        test_inputs, test_targets = batch['test']

        for task_idx, (input, target, test_input, test_target) in enumerate(
                zip(inputs, targets, test_inputs, test_targets)):
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                for step in range(num_adapt_steps):
                    inner_loss = inner_criterion(fmodel(input), target)
                    diffopt.step(inner_loss)
                test_logit = fmodel(test_input).detach()
                test_loss = outer_criterion(test_logit, test_target)
                test_losses.append(test_loss.detach())

        losses = sum(test_losses) / len(tepoch)
        tepoch.set_postfix(loss=losses)
    '''
    if min_val_loss > test_losses:  # save the train model
        min_val_loss = test_losses
        checkpoint = {'Epoch': epoch,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        
        torch.save(checkpoint, params["checkpoint_path"] + model_params["name"] + "/"
                   + params["dataset_name"] + "_" + str(epoch) + "_"
                   + str(min_val_loss) + '.pth')
        
    '''


def maml_test(tepoch, model, inner_criterion, outer_criterion, inner_optimizer, num_adapt_steps):
    model.train()
    mean_test_loss = torch.tensor(0., device='cuda:9')
    inference_array = []
    target_array = []
    for batch in tepoch:
        tepoch.set_description(f"test")

        batch['train'][0] = batch['train'][0].view(1, -1, 6, 36, 36)
        batch['test'][0] = batch['test'][0].view(1, -1, 6, 36, 36)
        batch['train'][1] = batch['train'][1].view(1, -1, 1)
        batch['test'][1] = batch['test'][1].view(1, -1, 1)

        inputs, targets = batch['train']
        test_inputs, test_targets = batch['test']

        for task_idx, (input, target, test_input, test_target) in enumerate(
                zip(inputs, targets, test_inputs, test_targets)):
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                for step in range(num_adapt_steps):
                    inner_loss = inner_criterion(fmodel(input), target)
                    diffopt.step(inner_loss)
                test_logit = fmodel(test_input).detach()
                test_loss = outer_criterion(test_logit, test_target)
                mean_test_loss += test_loss.item()

                inference_array.extend(test_logit.cpu().numpy())
                target_array.extend(test_target.cpu().numpy())

    inference_array = detrend(np.cumsum(inference_array), 100)
    target_array = detrend(np.cumsum(target_array), 100)
    plot_graph(0, 300, target_array, inference_array)