import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size=3, hidden_size=128):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, input_size=3, hidden_size=128, output_size=360):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.length = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, hidden_size=128, input_size=3, output_size=1, label='ple'):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size
        )

        if label == 'abp':
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('encoder is frozen')

        self.decoder = Decoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        reconstruct_output, hidden = self.decoder(x, self.encoder(x))

        return reconstruct_output.view(x.shape[0], -1)
