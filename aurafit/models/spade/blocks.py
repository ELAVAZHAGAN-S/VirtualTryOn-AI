from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from aurafit.models.spade.norm import SPADENorm

class SPADEResBlock(nn.Module):
    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        super(SPADEResBlock, self).__init__()
        self.param_opt = opt
        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False)

        subnorm_type = opt.norm_G
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        gen_semantic_nc = opt.gen_semantic_nc
        if use_mask_norm:
            subnorm_type = 'aliasmask'
            gen_semantic_nc = gen_semantic_nc + 1

        self.norm_0 = SPADENorm(opt, subnorm_type, input_nc, gen_semantic_nc)
        self.norm_1 = SPADENorm(opt, subnorm_type, middle_nc, gen_semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADENorm(opt, subnorm_type, input_nc, gen_semantic_nc)

        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg, misalign_mask):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, misalign_mask))
        else:
            return x

    def forward(self, x, seg, misalign_mask=None):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        x_s = self.shortcut(x, seg, misalign_mask)

        dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        output = x_s + dx
        return output

__all__ = [
    "SPADEResBlock",
]