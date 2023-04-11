import torch

class MLP(torch.nn.Module):
    def __init__(self, features, input_dim, output_dim, dropout):
        super(MLP, self).__init__()
        
        layers = [torch.nn.Linear(input_dim, features[0])]
        for in_features, out_features in zip(features, features[1:]):
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
        
        layers.append(torch.nn.Linear(features[-1], output_dim))
        
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.view(-1)