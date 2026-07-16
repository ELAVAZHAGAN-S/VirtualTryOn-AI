from __future__ import annotations
import os
import time
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from aurafit.checkpoint.io import load_checkpoint
from aurafit.data.cp_dataset_test import CPDatasetTest
from aurafit.data.loader import CPDataLoader
from aurafit.models.condition_generator import ConditionGenerator
from aurafit.models.discriminators import define_D
from aurafit.preprocessing.norm_const import D_logit
from aurafit.visualization.segmap import visualize_segmap

def test(opt: Any, test_loader: CPDataLoader, board: Any, tocg: ConditionGenerator, D: Optional[Any] = None) -> None:
    tocg.cuda()
    tocg.eval()
    if D is not None:
        D.cuda()
        D.eval()

    os.makedirs(os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1], opt.datamode, opt.datasetting, 'multi-task'), exist_ok=True)
    num = 0
    iter_start_time = time.time()
    if D is not None:
        D_score = []
    for inputs in test_loader.data_loader:

        c_paired = inputs['cloth'][opt.datasetting].cuda()
        cm_paired = inputs['cloth_mask'][opt.datasetting].cuda()
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

            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()

            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask

                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
            if D is not None:
                fake_segmap_softmax = F.softmax(fake_segmap, dim=1)
                pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
                score = D_logit(pred_segmap)
                score = (score / (1 - score)) / opt.norm_const
                print("prob0", score)
                for i in range(cm_paired.shape[0]):
                    name = inputs['c_name']['paired'][i].replace('.jpg', '.png')
                    D_score.append((name, score[i].item()))

            fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalign = fake_clothmask - warped_cm_onehot
            misalign[misalign < 0.0] = 0.0

        for i in range(c_paired.shape[0]):
            grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                            (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                            visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            save_image(grid, os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                             opt.datamode, opt.datasetting, 'multi-task',
                             (inputs['c_name']['paired'][i].split('.')[0] + '_' +
                              inputs['c_name']['unpaired'][i].split('.')[0] + '.png')))
        num += c_paired.shape[0]
        print(num)
    if D is not None:
        D_score.sort(key=lambda x: x[1], reverse=True)
        for name, score in D_score:
            f = open(os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                                opt.datamode, opt.datasetting, 'multi-task', 'rejection_prob.txt'), 'a')
            f.write(name + ' ' + str(score) + '\n')
            f.close()
    print(f"Test time {time.time() - iter_start_time}")


def run(opt: Any) -> None:
    test_dataset = CPDatasetTest(opt, variant="condition")
    test_loader = CPDataLoader(opt, test_dataset)

    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1], opt.datamode, opt.datasetting))

    input1_nc = 4
    input2_nc = opt.semantic_nc + 3
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    if not opt.D_checkpoint == '' and os.path.exists(opt.D_checkpoint):
        if opt.norm_const is None:
            raise NotImplementedError
        D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2=opt.Ddownx2, Ddropout=opt.Ddropout, n_layers_D=3, spectral=opt.spectral, num_D=opt.num_D)
    else:
        D = None
    load_checkpoint(tocg, opt.tocg_checkpoint)
    if not opt.D_checkpoint == '' and os.path.exists(opt.D_checkpoint):
        load_checkpoint(D, opt.D_checkpoint)
    test(opt, test_loader, board, tocg, D=D)

    print("Finished testing!")

__all__ = [
    "test",
    "run",
]