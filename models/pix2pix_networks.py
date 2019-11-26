import torch.nn as nn
import functools
import torch


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, num_filters=(128, 256, 512, 512), use_norm=False, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_filters (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """

        # different from ano_pred with norm here
        super().__init__()
        if use_norm:
            if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
                use_bias = norm_layer.func != nn.InstanceNorm2d
            else:
                use_bias = norm_layer != nn.InstanceNorm2d
        else:
            use_bias = True

        self.net = []
        self.net.append(nn.Conv2d(input_nc, num_filters[0], kernel_size=4, padding=2, stride=2, bias=use_bias))
        self.net.append(nn.LeakyReLU(0.2, True))
        if use_norm:
            for i in range(1, len(num_filters) - 1):
                self.net.extend([nn.Conv2d(num_filters[i - 1], num_filters[i], 4, 2, 2, bias=use_bias),
                                 nn.LeakyReLU(0.2, True),
                                 norm_layer(num_filters[i])])
        else:
            for i in range(1, len(num_filters) - 1):
                self.net.extend([nn.Conv2d(num_filters[i - 1], num_filters[i], 4, 2, 2, bias=use_bias),
                                 nn.LeakyReLU(0.2, True)])

        self.net.extend([nn.Conv2d(num_filters[-2], num_filters[-1], 4, 1, 2, bias=use_bias),
                         nn.LeakyReLU(0.2, True)])

        self.net.append(nn.Conv2d(num_filters[-1], 1, 4, 1, 2))
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        out = self.net(input)
        out = torch.sigmoid(out)
        return out
