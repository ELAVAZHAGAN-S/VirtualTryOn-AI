from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurafit.models.blocks import BaseNetwork
from aurafit.models.spade.blocks import SPADEResBlock

class SPADEGenerator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super(SPADEGenerator, self).__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers
        self.param_opt = opt
        self.sh, self.sw = self.compute_latent_vector_size(opt)

        nf = opt.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))

        self.head_0 = SPADEResBlock(opt, nf * 16, nf * 16, use_mask_norm=False)

        self.G_middle_0 = SPADEResBlock(opt, nf * 16 + 16, nf * 16, use_mask_norm=False)
        self.G_middle_1 = SPADEResBlock(opt, nf * 16 + 16, nf * 16, use_mask_norm=False)

        self.up_0 = SPADEResBlock(opt, nf * 16 + 16, nf * 8, use_mask_norm=False)
        self.up_1 = SPADEResBlock(opt, nf * 8 + 16, nf * 4, use_mask_norm=False)
        self.up_2 = SPADEResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = SPADEResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = SPADEResBlock(opt, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def compute_latent_vector_size(self, opt):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))

        sh = opt.fine_height // 2 ** num_up_layers
        sw = opt.fine_width // 2 ** num_up_layers
        return sh, sw

    def forward(self, x, seg):
        samples = [F.interpolate(x, size=(self.sh * 2 ** i, self.sw * 2 ** i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]

        x = self.head_0(features[0], seg)
        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg)

        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), 1), seg)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), 1), seg)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)

__all__ = [
    "SPADEGenerator",
]