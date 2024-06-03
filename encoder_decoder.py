import torch
from torch import nn
from buildingblocks import ConvLayer, ResidualBlocks

class StyleFeatureExtractor(nn.Module):
    def __init__(self, downsample_steps, input_channels, base_channels, style_dim, norm_type, activation_type, padding_type):
        super(StyleFeatureExtractor, self).__init__()
        layers = []
        layers.append(ConvLayer(input_channels, base_channels, 7, 1, 3, norm=norm_type, activation=activation_type, pad_type=padding_type))
        for _ in range(2):
            layers.append(ConvLayer(base_channels, 2 * base_channels, 4, 2, 1, norm=norm_type, activation=activation_type, pad_type=padding_type))
            base_channels *= 2
        for _ in range(downsample_steps - 2):
            layers.append(ConvLayer(base_channels, base_channels, 4, 2, 1, norm=norm_type, activation=activation_type, pad_type=padding_type))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(base_channels, style_dim, 1, 1, 0))
        self.model = nn.Sequential(*layers)
        self.output_dim = base_channels

    def forward(self, x):
        return self.model(x)

class ContentFeatureExtractor(nn.Module):
    def __init__(self, downsample_steps, res_blocks, input_channels, base_channels, norm_type, activation_type, padding_type):
        super(ContentFeatureExtractor, self).__init__()
        layers = []
        layers.append(ConvLayer(input_channels, base_channels, 7, 1, 3, norm=norm_type, activation=activation_type, pad_type=padding_type))
        for _ in range(downsample_steps):
            layers.append(ConvLayer(base_channels, 2 * base_channels, 4, 2, 1, norm=norm_type, activation=activation_type, pad_type=padding_type))
            base_channels *= 2
        layers.append(ResidualBlocks(res_blocks, base_channels, norm=norm_type, activation=activation_type, pad_type=padding_type))
        self.model = nn.Sequential(*layers)
        self.output_dim = base_channels

    def forward(self, x):
        return self.model(x)

class ImageDecoder(nn.Module):
    def __init__(self, upsample_steps, res_blocks, base_channels, output_channels, res_norm='adain', activation='relu', padding='zero'):
        super(ImageDecoder, self).__init__()
        layers = []
        layers.append(ResidualBlocks(res_blocks, base_channels, res_norm, activation, pad_type=padding))
        for _ in range(upsample_steps):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(ConvLayer(base_channels, base_channels // 2, 5, 1, 2, norm='ln', activation=activation, pad_type=padding))
            base_channels //= 2
        layers.append(ConvLayer(base_channels, output_channels, 7, 1, 3, norm='none', activation='tanh', pad_type=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

