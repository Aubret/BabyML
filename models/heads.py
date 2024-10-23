import torch
from torch import nn


class ActionMultiLayerProj(nn.Module):
    def __init__(self, n, in_feature, hidden_feature, out_feature, *args, bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(n):
            in_f = in_feature if i == 0 else hidden_feature
            layers.append(nn.Linear(in_f, hidden_feature, bias=bias))
            layers.append(nn.BatchNorm1d(hidden_feature))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_feature, out_feature))
        self.layers = nn.Sequential(*layers)

    def forward_all(self, x, y):
        return self.layers(torch.cat((x, y), dim=1))

    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        return x

class MultiLayerProj(nn.Module):
    def __init__(self, n, in_feature, hidden_feature, out_feature, *args, bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(n):
            in_f = in_feature if i == 0 else hidden_feature
            layers.append(nn.Linear(in_f, hidden_feature, bias=bias))
            layers.append(nn.BatchNorm1d(hidden_feature))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_feature, out_feature))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)

class MultiLayerProjShortcut(nn.Module):
    def __init__(self, n, in_feature, hidden_feature, out_feature, *args, bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(n):
            in_f = in_feature if i == 0 else hidden_feature
            out_f = out_feature if i == n-1 else hidden_feature
            layers.append(nn.Linear(in_f, out_f, bias=bias))
            layers.append(nn.BatchNorm1d(out_f))
            if i != n-1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.layers(x)
        return self.final_relu(out + x[:, :out.shape[1]])
