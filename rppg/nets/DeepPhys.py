import torch

device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

class DeepPhys(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.attention_mask1 = None
        self.attention_mask2 = None

        self.appearance_model = AppearanceModel(in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=self.kernel_size)
        self.motion_model = MotionModel(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=self.kernel_size)

        self.linear_model = LinearModel()

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

class AppearanceModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        # Appearance model
        super().__init__()
        #1
        self.a_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        #2
        self.a_batch_Normalization1 = torch.nn.BatchNorm2d(out_channels)
        #3
        self.a_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                       padding=1)
        self.a_batch_Normalization2 = torch.nn.BatchNorm2d(out_channels)
        #drop 안됨
        self.a_dropout1 = torch.nn.Dropout2d(p=0.50)
        # Attention mask1
        self.attention_mask1 = AttentionBlock(out_channels)
        self.a_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1,
                                       padding=1)
        self.a_Batch_Normalization3 = torch.nn.BatchNorm2d(out_channels * 2)
        self.a_conv4 = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3,
                                       stride=1, padding=1)
        self.a_Batch_Normalization4 = torch.nn.BatchNorm2d(out_channels * 2)
        self.a_dropout2 = torch.nn.Dropout2d(p=0.50)
        # Attention mask2
        self.attention_mask2 = AttentionBlock(out_channels * 2)

    def forward(self, inputs):
        # Convolution layer
        A1 = torch.tanh(self.a_batch_Normalization1(self.a_conv1(inputs)))
        A2 = torch.tanh(self.a_batch_Normalization2(self.a_conv2(A1)))
        A3 = self.a_dropout1(A2)
        # Calculate Mask1
        M1 = self.attention_mask1(A3)
        # Pooling
        A4 = self.a_avg1(A3)
        # Convolution layer
        A5 = torch.tanh(self.a_Batch_Normalization3(self.a_conv3(A4)))
        A6 = torch.tanh(self.a_Batch_Normalization4(self.a_conv4(A5)))
        A7 = self.a_dropout2(A6)
        # Calculate Mask2
        M2 = self.attention_mask2(A7)

        return M1, M2


class MotionModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Motion model
        self.m_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization1 = torch.nn.BatchNorm2d(out_channels)
        self.m_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization2 = torch.nn.BatchNorm2d(out_channels)
        self.m_dropout1 = torch.nn.Dropout2d(p=0.50)

        self.m_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=kernel_size,
                                       stride=1,
                                       padding=1)
        self.m_batch_Normalization3 = torch.nn.BatchNorm2d(out_channels * 2)
        self.m_conv4 = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.m_batch_Normalization4 = torch.nn.BatchNorm2d(out_channels * 2)
        self.m_dropout2 = torch.nn.Dropout2d(p=0.50)
        self.m_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask1, mask2):
        M1 = torch.tanh(self.m_batch_Normalization1(self.m_conv1(inputs)))
        M2 = self.m_batch_Normalization2(self.m_conv2(M1))
        # element wise multiplication Mask1
        ones = torch.ones(size=M2.shape).to(device)
        g1 = torch.tanh(torch.mul(ones @ mask1, M2))
        M3 = self.m_dropout1(g1)
        # pooling
        M4 = self.m_avg1(M3)
        # g1 = torch.tanh(torch.mul(1 * mask1, M4))
        M5 = torch.tanh(self.m_batch_Normalization3(self.m_conv3(M4)))
        M6 = self.m_batch_Normalization4(self.m_conv4(M5))
        # element wise multiplication Mask2
        g2 = torch.tanh(torch.mul(1 * mask2, M6))
        M7 = self.m_dropout2(g2)
        M8 = self.m_avg2(M7)
        out = torch.tanh(M8)

        return out

class TSM_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.tsm1 = TSM()
        self.t_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       padding=padding)

    def forward(self, input, n_frame=2, fold_div=3):
        t = self.tsm1(input, n_frame, fold_div)
        t = self.t_conv1(t)
        return t


class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, n_frame=4, fold_div=3):
        n_frame = 4
        B, C, H, W = input.shape
        input = input.view(-1, n_frame, H, W, C)
        fold = C // fold_div
        last_fold = C - (fold_div - 1) * fold
        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], -1)

        padding1 = torch.zeros_like(out1)
        padding1 = padding1[:, -1, :, :, :]
        padding1 = torch.unsqueeze(padding1, 1)
        _, out1 = torch.split(out1, [1, n_frame - 1], 1)
        out1 = torch.cat((out1, padding1), 1)

        padding2 = torch.zeros_like(out2)
        padding2 = padding2[:, 0, :, :, :]
        padding2 = torch.unsqueeze(padding2, 1)
        out2, _ = torch.split(out2, [n_frame - 1, 1], 1)
        out2 = torch.cat((padding2, out2), 1)

        out = torch.cat((out1, out2, out3), -1)
        out = out.view([-1, C, H, W])

        return out

class LinearModel(torch.nn.Module):
    def __init__(self, in_channel=5184):
        super().__init__()
        self.f_drop1 = torch.nn.Dropout(0.25)
        self.f_linear1 = torch.nn.Linear(in_channel, 256, bias=True)
        self.f_linear2 = torch.nn.Linear(256, 1, bias=True)

    def forward(self, input):
        f1 = torch.flatten(input, start_dim=1)
        f2 = self.f_drop1(f1)
        f3 = torch.tanh(self.f_linear1(f2))
        f4 = self.f_linear2(f3)
        return f4

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = input.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        return mask
