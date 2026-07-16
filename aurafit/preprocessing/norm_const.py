from __future__ import annotations
import os
import time
from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurafit.checkpoint.io import load_checkpoint
from aurafit.data.cp_dataset import CPDataset
from aurafit.data.loader import CPDataLoader
from aurafit.models.condition_generator import ConditionGenerator
from aurafit.models.discriminators import define_D

def D_logit(pred: Any) -> torch.Tensor:
    score = 0
    for i in pred:
        score += i[-1].mean((1, 2, 3)) / 2
    return score


def get_const(opt: Any, train_loader: CPDataLoader, tocg: ConditionGenerator, D: Any, length: int) -> float:
    D.cuda()
    D.eval()
    tocg.cuda()
    tocg.eval()

    logit_list: List[float] = []
    i = 0
    for step in range(length // opt.batch_size):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        label_onehot = inputs['parse_onehot'].cuda()
        label = inputs['parse'].cuda()
        parse_cloth_mask = inputs['pcm'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im = inputs['image']
        with torch.no_grad():
            input1 = torch.cat([c_paired, cm_paired], 1)
            input2 = torch.cat([parse_agnostic, densepose], 1)

            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)
            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                    cloth_mask = torch.ones_like(fake_segmap.detach())
                    cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask

                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap.detach())
                    cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask

            fake_segmap_softmax = F.softmax(fake_segmap, dim=1)

            real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label), dim=1))
            fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))

            print("real:", D_logit(real_segmap_pred), "fake:", D_logit(fake_segmap_pred))
            logit_real = D_logit(real_segmap_pred)
            logit_fake = D_logit(fake_segmap_pred)
            for l in logit_real:
                l = l / (1 - l)
                logit_list.append(l.item())
            for l in logit_fake:
                l = l / (1 - l)
                logit_list.append(l.item())

        print("i:", i)
    logit_list.sort()

    return logit_list[-1]


def run(opt: Any) -> None:
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    input1_nc = 4
    input2_nc = opt.semantic_nc + 3
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2=opt.Ddownx2, Ddropout=opt.Ddropout, n_layers_D=3, spectral=opt.spectral, num_D=opt.num_D)
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    load_checkpoint(D, opt.D_checkpoint)
    load_checkpoint(tocg, opt.tocg_checkpoint)

    M = get_const(opt, train_loader, tocg, D, length=len(train_dataset))
    print("M:", M)

__all__ = [
    "D_logit",
    "get_const",
    "run",
]