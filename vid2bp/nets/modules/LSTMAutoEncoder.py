import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, length=360, hidden_size=128, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(length, hidden_size, num_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, length=360, hidden_size=128, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.length = length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(length, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, length)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, hidden_size=128, length=360, num_layers=1, label='ple'):
        super(LSTMAutoEncoder, self).__init__()
        self.length = length
        self.encoder = Encoder(
            length=length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        if label == 'abp':
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('encoder is frozen')

        self.decoder = Decoder(
            length=length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, x):
        encoder_hidden = self.encoder(x)

        hidden = encoder_hidden
        for t in range(self.length):
            reconstruct_output, hidden = self.decoder(x, hidden)

        return reconstruct_output
