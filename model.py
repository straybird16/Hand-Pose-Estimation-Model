from torch.nn.modules.activation import Softmax
from torch.nn.modules.dropout import Dropout
# ---------------------------------------------------------
# Step 5. Create Model
# ---------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm

# This helper function is from Assignment 3 in CSC 674, UMASS Amherst, Spring 2022
# It creates a multi-layer perceptron (consists of multiple layers of nn.Linear)
# with specified layer construction
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i] // 32)
            else:
                num_groups.append(1)
        return Seq(*[
                    Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2)
                        )
                    for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features
# num_output_features: number of output per-point descriptors (23, which is 22 joints + none category)
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features=3, num_output_features=256):
        super(PointNet, self).__init__()
        self.input_features = num_input_features
        self.output_features = num_output_features
        self.num_points = 600
        self.mlp = MLP([num_input_features, 32, num_output_features])
        self.featureExtractionLayer = Seq(
            # self.T_net,
            # self.feature_transform,
            self.mlp
        )

    def forward(self, x):
        N = x.shape[-2]
        x = self.featureExtractionLayer(x)
        # x -> N x F = 600 x 256
        x = torch.max(x, -2, keepdim=True)[0]
        # x = self.fc(x)
        return x


class pnGroup(torch.nn.Module):
    # takes 20 x 256
    def __init__(self, num_input_features, num_output_features):
        super(pnGroup, self).__init__()
        self.out_num = num_output_features
        self.point_net = PointNet(num_input_features, num_output_features)

    def forward(self, x):
        y, x = x[0], x[1:]
        y = self.point_net(y)
        for frame in x:
            y = torch.cat((y, self.point_net(frame)), 0)

        return y


class PPN(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, device=torch.device('cuda:0'), hidden_size=32):
        super(PPN, self).__init__()
        self.device = device
        self.png = pnGroup(num_input_features, num_output_features)
        self.partial = False
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.num_points = 600
        self.input_size = num_output_features
        self.sequence_length = 20
        self.num_classes = 14
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0)
        self.fc = Seq(
            Lin(self.hidden_size * self.sequence_length, 128),
            nn.Dropout(p=0.4),
            Lin(128, self.num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        y, x = x[0], x[1:]
        y = self.png(y)
        for pc in x:
            y = torch.cat((y, self.png(pc)), 0)

        # y -> B x 20 x 32
        if not self.partial:
            # LSTM forward
            y = y.reshape(B, self.sequence_length, y.size(-1))
            h0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(self.device)
            out, _ = self.lstm(y, (h0, c0))
            out = out.reshape(out.shape[0], -1)
            out = self.fc(out)
            # print(out.shape)
            # out = nn.functional.normalize(out, dim=-1)
            return out

        else:
            y = y.reshape(B, self.sequence_length * y.size(-1))
            out = self.fc(y)
            return out
