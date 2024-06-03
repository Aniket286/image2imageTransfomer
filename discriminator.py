from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from buildingblocks import ConvLayer

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
        layers.append(ConvLayer(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activation, pad_type=self.padding_type))
        for _ in range(self.n_layers - 1):
            layers.append(ConvLayer(dim, dim * 2, 4, 2, 1, norm=self.normalization, activation=self.activation, pad_type=self.padding_type))
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
    

