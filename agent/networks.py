import torch


import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn.distributions import NormalParamExtractor

class ActorNet(nn.Module):
    def __init__(self, num_cells, input_size, hidden_size, output_size, device):
        super(ActorNet, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, device=device)
        self.fc = nn.Sequential(
            nn.LazyLinear(input_size, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(2 * output_size, device=device),
            NormalParamExtractor(),
        )

    def forward(self, x):
        # if len(x.size()) == 1:
        #     # If input is unbatched, add a batch dimension
        #     x = x.unsqueeze(0)
        # lstm_out, _ = self.lstm(x)
        x = self.fc(x)
        return x

class CriticNet(nn.Module):
    def __init__(self, num_cells, input_size, hidden_size, output_size, device):
        super(CriticNet, self).__init__()
        # self.lstm = self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, device=device)
        self.fc = nn.Sequential(
            nn.LazyLinear(input_size, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(output_size, device=device),
        )

    def forward(self, x):
        # if len(x.size()) == 1:
        #     # If input is unbatched, add a batch dimension
        #     x = x.unsqueeze(0)
        # lstm_out, _ = self.lstm(x)
        x = self.fc(x)
        return x