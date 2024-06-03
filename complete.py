from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

# Handle Python 2/3 compatibility for zip function
try:
    from itertools import izip as zip
except ImportError:
    pass

##################################################################################
# Discriminator
##################################################################################

class MultiScaleImageDiscriminator(nn.Module):
    # Architecture for multi-scale image discriminator
    def __init__(self, input_dim, config):
        super(MultiScaleImageDiscriminator, self).__init__()
        self.n_layers = config['n_layer']
        self.gan_type = config['gan_type']
        self.base_dim = config['dim']
        self.normalization = config['norm']
        self.activation = config['activ']
        self.num_scales = config['num_scales']
        self.padding_type = config['pad_type']
        self.input_dim = input_dim
        self.downsampler = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.scale_nets = nn.ModuleList()
        for _ in range(self.num_scales):
            self.scale_nets.append(self._create_network())

    def _create_network(self):
        dim = self.base_dim
        layers = []
        layers.append(Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activation, pad_type=self.padding_type))
        for _ in range(self.n_layers - 1):
            layers.append(Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.normalization, activation=self.activation, pad_type=self.padding_type))
            dim *= 2
        layers.append(nn.Conv2d(dim, 1, 1, 1, 0))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for scale_net in self.scale_nets:
            outputs.append(scale_net(x))
            x = self.downsampler(x)
        return outputs

    def compute_discriminator_loss(self, fake_input, real_input):
        # Compute the loss for training the discriminator
        fake_outputs = self.forward(fake_input)
        real_outputs = self.forward(real_input)
        total_loss = 0

        for fake_out, real_out in zip(fake_outputs, real_outputs):
            if self.gan_type == 'lsgan':
                total_loss += torch.mean((fake_out - 0) ** 2) + torch.mean((real_out - 1) ** 2)
            elif self.gan_type == 'nsgan':
                zeros = Variable(torch.zeros_like(fake_out.data).cuda(), requires_grad=False)
                ones = Variable(torch.ones_like(real_out.data).cuda(), requires_grad=False)
                total_loss += torch.mean(F.binary_cross_entropy(F.sigmoid(fake_out), zeros) +
                                         F.binary_cross_entropy(F.sigmoid(real_out), ones))
            else:
                raise ValueError("Unsupported GAN type: {}".format(self.gan_type))
        return total_loss

    def compute_generator_loss(self, fake_input):
        # Compute the loss for training the generator
        fake_outputs = self.forward(fake_input)
        total_loss = 0
        for fake_out in fake_outputs:
            if self.gan_type == 'lsgan':
                total_loss += torch.mean((fake_out - 1) ** 2)
            elif self.gan_type == 'nsgan':
                ones = Variable(torch.ones_like(fake_out.data).cuda(), requires_grad=False)
                total_loss += torch.mean(F.binary_cross_entropy(F.sigmoid(fake_out), ones))
            else:
                raise ValueError("Unsupported GAN type: {}".format(self.gan_type))
        return total_loss
    

import torch
from torch import nn
from torch.autograd import Variable

##################################################################################
# Generator
##################################################################################

class AdaptiveInstanceNormGenerator(nn.Module):
    # Adaptive Instance Normalization Auto-Encoder
    def __init__(self, input_dim, params):
        super(AdaptiveInstanceNormGenerator, self).__init__()
        self.dim = params['dim']
        self.style_dim = params['style_dim']
        self.num_downsample = params['n_downsample']
        self.num_res_blocks = params['n_res']
        self.activation = params['activ']
        self.padding_type = params['pad_type']
        self.mlp_dim = params['mlp_dim']

        # Initialize style encoder
        self.style_encoder = StyleEncoder(4, input_dim, self.dim, self.style_dim, norm='none', activ=self.activation, pad_type=self.padding_type)

        # Initialize content encoder
        self.content_encoder = ContentEncoder(self.num_downsample, self.num_res_blocks, input_dim, self.dim, 'in', self.activation, pad_type=self.padding_type)
        self.decoder = Decoder(self.num_downsample, self.num_res_blocks, self.content_encoder.output_dim, input_dim, res_norm='adain', activ=self.activation, pad_type=self.padding_type)

        # Initialize MLP for generating AdaIN parameters
        self.mlp = MLP(self.style_dim, self._calculate_adain_params(self.decoder), self.mlp_dim, 3, norm='none', activ=self.activation)

    def forward(self, x):
        # Reconstruct images
        content_code, style_code = self.encode(x)
        reconstructed_images = self.decode(content_code, style_code)
        return reconstructed_images

    def encode(self, x):
        # Encode image to content and style codes
        style_code = self.style_encoder(x)
        content_code = self.content_encoder(x)
        return content_code, style_code

    def decode(self, content, style):
        # Decode content and style codes to image
        adain_parameters = self.mlp(style)
        self._set_adain_params(adain_parameters, self.decoder)
        reconstructed_image = self.decoder(content)
        return reconstructed_image

    def _set_adain_params(self, adain_params, model):
        # Assign AdaIN parameters to corresponding layers
        for module in model.modules():
            if module.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :module.num_features]
                std = adain_params[:, module.num_features:2 * module.num_features]
                module.bias = mean.contiguous().view(-1)
                module.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * module.num_features:
                    adain_params = adain_params[:, 2 * module.num_features:]

    def _calculate_adain_params(self, model):
        # Calculate the number of AdaIN parameters needed
        total_adain_params = 0
        for module in model.modules():
            if module.__class__.__name__ == "AdaptiveInstanceNorm2d":
                total_adain_params += 2 * module.num_features
        return total_adain_params


class VariationalAutoencoderGenerator(nn.Module):
    # Variational Autoencoder (VAE) Architecture
    def __init__(self, input_dim, params):
        super(VariationalAutoencoderGenerator, self).__init__()
        self.dim = params['dim']
        self.num_downsample = params['n_downsample']
        self.num_res_blocks = params['n_res']
        self.activation = params['activ']
        self.padding_type = params['pad_type']

        # Initialize content encoder
        self.encoder = ContentEncoder(self.num_downsample, self.num_res_blocks, input_dim, self.dim, 'in', self.activation, pad_type=self.padding_type)
        self.decoder = Decoder(self.num_downsample, self.num_res_blocks, self.encoder.output_dim, input_dim, res_norm='in', activ=self.activation, pad_type=self.padding_type)

    def forward(self, x):
        # Forward pass through VAE
        encoded_hiddens = self.encode(x)
        if self.training:
            noise = Variable(torch.randn(encoded_hiddens.size()).cuda(encoded_hiddens.data.get_device()))
            reconstructed_images = self.decode(encoded_hiddens + noise)
        else:
            reconstructed_images = self.decode(encoded_hiddens)
        return reconstructed_images, encoded_hiddens

    def encode(self, x):
        hiddens = self.encoder(x)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        decoded_images = self.decoder(hiddens)
        return decoded_images

import torch
from torch import nn

##################################################################################
# Encoder and Decoders
##################################################################################

class StyleFeatureExtractor(nn.Module):
    def __init__(self, downsample_steps, input_channels, base_channels, style_dim, norm_type, activation_type, padding_type):
        super(StyleFeatureExtractor, self).__init__()
        layers = []
        layers.append(Conv2dBlock(input_channels, base_channels, 7, 1, 3, norm=norm_type, activation=activation_type, pad_type=padding_type))
        for _ in range(2):
            layers.append(Conv2dBlock(base_channels, 2 * base_channels, 4, 2, 1, norm=norm_type, activation=activation_type, pad_type=padding_type))
            base_channels *= 2
        for _ in range(downsample_steps - 2):
            layers.append(Conv2dBlock(base_channels, base_channels, 4, 2, 1, norm=norm_type, activation=activation_type, pad_type=padding_type))
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
        layers.append(Conv2dBlock(input_channels, base_channels, 7, 1, 3, norm=norm_type, activation=activation_type, pad_type=padding_type))
        for _ in range(downsample_steps):
            layers.append(Conv2dBlock(base_channels, 2 * base_channels, 4, 2, 1, norm=norm_type, activation=activation_type, pad_type=padding_type))
            base_channels *= 2
        layers.append(ResBlocks(res_blocks, base_channels, norm=norm_type, activation=activation_type, pad_type=padding_type))
        self.model = nn.Sequential(*layers)
        self.output_dim = base_channels

    def forward(self, x):
        return self.model(x)

class ImageDecoder(nn.Module):
    def __init__(self, upsample_steps, res_blocks, base_channels, output_channels, res_norm='adain', activation='relu', padding='zero'):
        super(ImageDecoder, self).__init__()
        layers = []
        layers.append(ResBlocks(res_blocks, base_channels, res_norm, activation, pad_type=padding))
        for _ in range(upsample_steps):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(Conv2dBlock(base_channels, base_channels // 2, 5, 1, 2, norm='ln', activation=activation, pad_type=padding))
            base_channels //= 2
        layers.append(Conv2dBlock(base_channels, output_channels, 7, 1, 3, norm='none', activation='tanh', pad_type=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################
class ResidualBlocks(nn.Module):
    def __init__(self, num_blocks, base_channels, norm_type='in', activation_type='relu', padding_type='zero'):
        super(ResidualBlocks, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels, norm=norm_type, activation=activation_type, pad_type=padding_type))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FullyConnectedLayers(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels, num_blocks, norm_type='none', activation_type='relu'):
        super(FullyConnectedLayers, self).__init__()
        layers = []
        layers.append(LinearBlock(input_channels, base_channels, norm=norm_type, activation=activation_type))
        for _ in range(num_blocks - 2):
            layers.append(LinearBlock(base_channels, base_channels, norm=norm_type, activation=activation_type))
        layers.append(LinearBlock(base_channels, output_channels, norm='none', activation='none'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    

##################################################################################
# Basic Building Blocks
##################################################################################

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

import torch.nn as nn
import torch.nn.functional as F

##################################################################################
# VGG Network Redefined
##################################################################################

class VGG16Modified(nn.Module):
    def __init__(self):
        super(VGG16Modified, self).__init__()
        self.layer1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.layer2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.layer3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.layer4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.layer5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.layer1_1(x), inplace=True)
        x = F.relu(self.layer1_2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.layer2_1(x), inplace=True)
        x = F.relu(self.layer2_2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.layer3_1(x), inplace=True)
        x = F.relu(self.layer3_2(x), inplace=True)
        x = F.relu(self.layer3_3(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.layer4_1(x), inplace=True)
        x = F.relu(self.layer4_2(x), inplace=True)
        x = F.relu(self.layer4_3(x), inplace=True)

        x = F.relu(self.layer5_1(x), inplace=True)
        x = F.relu(self.layer5_2(x), inplace=True)
        x = F.relu(self.layer5_3(x), inplace=True)
        output = x

        return output
    
    import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################################################
# Normalization Layers
##################################################################################

class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('mean_buffer', torch.zeros(num_features))
        self.register_buffer('var_buffer', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Weights and bias must be set before calling!"
        batch_size, channels = x.size(0), x.size(1)
        mean_rep = self.mean_buffer.repeat(batch_size)
        var_rep = self.var_buffer.repeat(batch_size)

        reshaped_x = x.contiguous().view(1, batch_size * channels, *x.size()[2:])
        normalized = F.batch_norm(
            reshaped_x, mean_rep, var_rep, self.weight, self.bias, True, self.momentum, self.eps
        )

        return normalized.view(batch_size, channels, *x.size()[2:])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features})"

class LayerNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        normalized_x = (x - mean) / (std + self.eps)

        if self.affine:
            gamma_shape = [1, -1] + [1] * (x.dim() - 2)
            normalized_x = normalized_x * self.gamma.view(*gamma_shape) + self.beta.view(*gamma_shape)

        return normalized_x

def l2_normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNormalization(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNormalization, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._params_created():
            self._create_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2_normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2_normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _params_created(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _create_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2_normalize(u.data)
        v.data = l2_normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *inputs):
        self._update_u_v()
        return self.module.forward(*inputs)

