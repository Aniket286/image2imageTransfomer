import torch
from torch import nn
from normalization import LayerNorm, AdaptiveInstanceNorm2d, SpectralNorm

class ResidualBlocks(nn.Module):
    def __init__(self, num_blocks, base_channels, norm_type='in', activation_type='relu', padding_type='zero'):
        super(ResidualBlocks, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels, norm=norm_type, activation=activation_type, pad_type=padding_type))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FullyConnectedLayers(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels, num_blocks, norm_type='none', activation_type='relu'):
        super(FullyConnectedLayers, self).__init__()
        layers = []
        layers.append(LinearLayer(input_channels, base_channels, norm=norm_type, activation=activation_type))
        for _ in range(num_blocks - 2):
            layers.append(LinearLayer(base_channels, base_channels, norm=norm_type, activation=activation_type))
        layers.append(LinearLayer(base_channels, output_channels, norm='none', activation='none'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_type='in', activation_type='relu', padding_type='zero'):
        super(ResidualBlock, self).__init__()
        layers = [
            ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, norm_type=norm_type, activation_type=activation_type, padding_type=padding_type),
            ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, norm_type=norm_type, activation_type='none', padding_type=padding_type)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm_type='none', activation_type='relu', padding_type='zero'):
        super(ConvLayer, self).__init__()
        self.use_bias = True

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding)
        elif padding_type == 'replicate':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_type == 'zero':
            self.padding = nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {padding_type}")

        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm_type == 'adain':
            self.norm = AdaptiveInstanceNorm2d(out_channels)
        elif norm_type in ['none', 'sn']:
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == 'prelu':
            self.activation = nn.PReLU()
        elif activation_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        if norm_type == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.padding(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, norm_type='none', activation_type='relu'):
        super(LinearLayer, self).__init__()
        use_bias = True

        if norm_type == 'sn':
            self.fc = SpectralNorm(nn.Linear(in_features, out_features, bias=use_bias))
        else:
            self.fc = nn.Linear(in_features, out_features, bias=use_bias)

        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(out_features)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm1d(out_features)
        elif norm_type == 'ln':
            self.norm = LayerNorm(out_features)
        elif norm_type in ['none', 'sn']:
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == 'prelu':
            self.activation = nn.PReLU()
        elif activation_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
