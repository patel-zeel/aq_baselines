import torch

class Encoder(torch.nn.Module):
    def __init__(self, features, input_dim, output_dim, encoding_dim, dropout):
        super(Encoder, self).__init__()
        
        layers = [torch.nn.Linear(input_dim+output_dim, features[0])]
        for in_features, out_features in zip(features, features[1:]):
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
        
        layers.append(torch.nn.Linear(features[-1], encoding_dim))
        
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_context, y_context):
        x = torch.cat([x_context, y_context.view(x_context.shape[0], -1)], dim=1)
        for layer in self.layers:
            x = layer(x)
            
        x = torch.mean(x, dim=0, keepdim=True)
        return x # (1, encoding_dim)
    
    
class Decoder(torch.nn.Module):
    def __init__(self, features, input_dim, encoding_dim, dropout):
        super(Decoder, self).__init__()
        
        layers = [torch.nn.Linear(encoding_dim+input_dim, features[0])]
        for in_features, out_features in zip(features, features[1:]):
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
        
        layers.append(torch.nn.Linear(features[-1], 2))
        
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_target, representation):
        x = torch.cat([x_target, representation.repeat(x_target.shape[0], 1)], dim=1)
        for layer in self.layers:
            x = layer(x)
        loc, log_scale = x[:, 0], x[:, 1]
        
        scale = torch.exp(log_scale)
        return loc, scale
    
class CNP(torch.nn.Module):
    def __init__(self, encoder_features, decoder_features, input_dim, output_dim, encoding_dim, dropout):
        super(CNP, self).__init__()
        
        self.encoder = Encoder(encoder_features, input_dim, output_dim, encoding_dim, dropout)
        self.decoder = Decoder(decoder_features, input_dim, encoding_dim, dropout)
        
    def forward(self, x_context, y_context, x_target):
        representation = self.encoder(x_context, y_context)
        loc, scale = self.decoder(x_target, representation)
        return loc, scale
        
# write a test
if __name__ == "__main__":
    x_context = torch.randn(10, 2)
    y_context = torch.randn(10, 1)
    x_target = torch.randn(15, 2)
    
    model = CNP(encoder_features=[32, 8], decoder_features=[16, 4], input_dim=2, output_dim=1, encoding_dim=3, dropout=0.1)
    loc, scale = model(x_context, y_context, x_target)
    print(loc.shape, scale.shape)