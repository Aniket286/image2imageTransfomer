import torch
from torch import nn
from torch.autograd import Variable
from encoder_decoder import StyleFeatureExtractor, ContentFeatureExtractor, ImageDecoder
from buildingblocks import FullyConnectedLayers

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
        self.style_encoder = StyleFeatureExtractor(4, input_dim, self.dim, self.style_dim, norm='none', activ=self.activation, pad_type=self.padding_type)

        # Initialize content encoder
        self.content_encoder = ContentFeatureExtractor(self.num_downsample, self.num_res_blocks, input_dim, self.dim, 'in', self.activation, pad_type=self.padding_type)
        self.decoder = ImageDecoder(self.num_downsample, self.num_res_blocks, self.content_encoder.output_dim, input_dim, res_norm='adain', activ=self.activation, pad_type=self.padding_type)

        # Initialize MLP for generating AdaIN parameters
        self.mlp = FullyConnectedLayers(self.style_dim, self._calculate_adain_params(self.decoder), self.mlp_dim, 3, norm='none', activ=self.activation)

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
        self.encoder = ContentFeatureExtractor(self.num_downsample, self.num_res_blocks, input_dim, self.dim, 'in', self.activation, pad_type=self.padding_type)
        self.decoder = ImageDecoder(self.num_downsample, self.num_res_blocks, self.encoder.output_dim, input_dim, res_norm='in', activ=self.activation, pad_type=self.padding_type)

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
