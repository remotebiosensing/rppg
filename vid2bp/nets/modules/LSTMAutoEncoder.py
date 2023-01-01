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
        self.lstm = nn.LSTM(length, hidden_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=False)
        self.fc = nn.Linear(hidden_size, length)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, hidden_size, length, num_layers):
        super(LSTMAutoEncoder, self).__init__()

        self.encoder = Encoder(
            length=length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.decoder = Decoder(
            length=length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, x):
        encoder_hidden = self.encoder(x)

        reconstruct_output = []
        hidden = encoder_hidden
        for t in range(self.length):
            temp_input, hidden = self.decoder(x, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)

        return reconstruct_output
