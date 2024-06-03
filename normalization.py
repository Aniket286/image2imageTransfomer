import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
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

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
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

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
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
