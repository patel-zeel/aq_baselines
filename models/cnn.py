import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, features, input_dim, kernel_size, output_dim, dropout):
        super(CNN, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=kernel_size, padding=padding)
        
        layers = []
        layers.append(nn.Linear(input_dim*2, features[0]))
        for in_features, out_features in zip(features, features[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.Linear(features[-1], output_dim))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = F.dropout(F.relu(self.conv(x)))
        x = x.permute(2,0,1).reshape(-1, 46)
        
        for layer in self.layers:
            x = layer(x)
        return x.view(-1)