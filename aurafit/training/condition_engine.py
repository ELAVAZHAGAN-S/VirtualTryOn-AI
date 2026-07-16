from __future__ import annotations
import os
import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision.utils import make_grid
from aurafit.checkpoint.io import load_checkpoint, save_checkpoint
from aurafit.data.cp_dataset import CPDataset
from aurafit.data.cp_dataset_test import CPDatasetTest
from aurafit.data.loader import CPDataLoader
from aurafit.inference.postprocess import remove_overlap
from aurafit.models.condition_generator import ConditionGenerator
from aurafit.models.discriminators import define_D
from aurafit.models.losses import GANLossVanilla, VGGLoss, cross_entropy2d
from aurafit.training.engine import (
    create_summary_writer,
    is_display_step,
    is_save_step,
    run_training_loop,
)
from aurafit.training.metrics import iou_metric
from aurafit.utils.grid import make_grid as mkgrid
from aurafit.visualization.segmap import visualize_segmap

def train(opt: Any, train_loader: CPDataLoader, test_loader: CPDataLoader, val_loader: CPDataLoader, board: Any, tocg: ConditionGenerator, D: Any) -> None:
    tocg.cuda()
    tocg.train()
    D.cuda()
    D.train()

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    if opt.fp16:
        criterionGAN = GANLossVanilla(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else:
        criterionGAN = GANLossVanilla(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)

    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))

    def step_fn(step: int) -> None:
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

        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)

        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        if opt.clothmask_composition != 'no_composition':
            if opt.clothmask_composition == 'detach':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                fake_segmap = fake_segmap * cloth_mask

            if opt.clothmask_composition == 'warp_grad':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                fake_segmap = fake_segmap * cloth_mask
        if opt.occlusion:
            warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
            warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1 - warped_clothmask_paired)

        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_cm_onehot
        misalign[misalign < 0.0] = 0.0

        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0

        if opt.edgeawaretv == 'no_edge':
            if not opt.lasttvonly:
                for flow in flow_list:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            else:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
        else:
            if opt.edgeawaretv == 'last_only':
                flow = flow_list[-1]
                warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                mask_y = torch.exp(-150 * torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                mask_x = torch.exp(-150 * torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                y_tv = y_tv * mask_y
                x_tv = x_tv * mask_x
                y_tv = y_tv.mean()
                x_tv = x_tv.mean()
                loss_tv = loss_tv + y_tv + x_tv

            elif opt.edgeawaretv == 'weighted':
                for i in range(5):
                    flow = flow_list[i]
                    warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                    mask_y = torch.exp(-150 * torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                    mask_x = torch.exp(-150 * torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                    y_tv = y_tv * mask_y
                    x_tv = x_tv * mask_x
                    y_tv = y_tv.mean() / (2 ** (4 - i))
                    x_tv = x_tv.mean() / (2 ** (4 - i))
                    loss_tv = loss_tv + y_tv + x_tv

            if opt.add_lasttv:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv

        N, _, iH, iW = c_paired.size()
        if opt.interflowloss:
            for i in range(len(flow_list) - 1):
                flow = flow_list[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size=c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)
                loss_l1_cloth += criterionL1(warped_cm, parse_cloth_mask) / (2 ** (4 - i))
                loss_vgg += criterionVGG(warped_c, im_c) / (2 ** (4 - i))

        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())

        if opt.no_GAN_loss:
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        else:
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)
            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            loss_G_GAN = criterionGAN(pred_segmap, True)

            if not opt.G_D_seperate:
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()), dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label), dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)

                loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)
                loss_D = loss_D_fake + loss_D_real

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            else:
                loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                with torch.no_grad():
                    _, fake_segmap, _, _ = tocg(input1, input2)
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)

                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()), dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label), dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)

                loss_D = loss_D_fake + loss_D_real

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
        if (step + 1) % opt.val_count == 0:
            tocg.eval()
            iou_list = []
            with torch.no_grad():
                for cnt in range(2000 // opt.batch_size):

                    inputs = val_loader.next_batch()
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

                    input1 = torch.cat([c_paired, cm_paired], 1)
                    input2 = torch.cat([parse_agnostic, densepose], 1)

                    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)

                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap.detach())
                            cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask

                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap.detach())
                            cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask

                    iou = iou_metric(F.softmax(fake_segmap, dim=1).detach(), label)
                    iou_list.append(iou.item())

            tocg.train()
            board.add_scalar('val/iou', np.mean(iou_list), step + 1)

        if (step + 1) % opt.tensorboard_count == 0:
            board.add_scalar('Loss/G', loss_G.item(), step + 1)
            board.add_scalar('Loss/G/l1_cloth', loss_l1_cloth.item(), step + 1)
            board.add_scalar('Loss/G/vgg', loss_vgg.item(), step + 1)
            board.add_scalar('Loss/G/tv', loss_tv.item(), step + 1)
            board.add_scalar('Loss/G/CE', CE_loss.item(), step + 1)
            if not opt.no_GAN_loss:
                board.add_scalar('Loss/G/GAN', loss_G_GAN.item(), step + 1)
                board.add_scalar('Loss/D', loss_D.item(), step + 1)
                board.add_scalar('Loss/D/pred_real', loss_D_real.item(), step + 1)
                board.add_scalar('Loss/D/pred_fake', loss_D_fake.item(), step + 1)

            grid = make_grid([(c_paired[0].cpu() / 2 + 0.5), (cm_paired[0].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu()), ((densepose.cpu()[0] + 1) / 2),
                              (im_c[0].cpu() / 2 + 0.5), parse_cloth_mask[0].cpu().expand(3, -1, -1), (warped_cloth_paired[0].cpu().detach() / 2 + 0.5), (warped_cm_onehot[0].cpu().detach()).expand(3, -1, -1),
                              visualize_segmap(label.cpu()), visualize_segmap(fake_segmap.cpu()), (im[0] / 2 + 0.5), (misalign[0].cpu().detach()).expand(3, -1, -1)],
                              nrow=4)
            board.add_images('train_images', grid.unsqueeze(0), step + 1)

            if not opt.no_test_visualize:
                inputs = test_loader.next_batch()
                c_paired = inputs['cloth'][opt.test_datasetting].cuda()
                cm_paired = inputs['cloth_mask'][opt.test_datasetting].cuda()
                cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                parse_agnostic = inputs['parse_agnostic'].cuda()
                densepose = inputs['densepose'].cuda()
                openpose = inputs['pose'].cuda()
                label_onehot = inputs['parse_onehot'].cuda()
                label = inputs['parse'].cuda()
                parse_cloth_mask = inputs['pcm'].cuda()
                im_c = inputs['parse_cloth'].cuda()
                im = inputs['image']

                tocg.eval()
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
                    if opt.occlusion:
                        warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
                        warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1 - warped_clothmask_paired)

                    fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
                    misalign = fake_clothmask - warped_cm_onehot
                    misalign[misalign < 0.0] = 0.0

                for i in range(opt.num_test_visualize):
                    grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i] + 1) / 2),
                                      (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                                      visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i] / 2 + 0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                      nrow=4)
                    board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                tocg.train()

        if is_display_step(step, opt):
            t = time.time() - iter_start_time
            if not opt.no_GAN_loss:
                print("step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV loss: %.4f CE: %.4f, G GAN: %.4f\nloss D: %.4f, D real: %.4f, D fake: %.4f" % (step + 1, t, loss_G.item(), loss_l1_cloth.item(), loss_vgg.item(), loss_tv.item(), CE_loss.item(), loss_G_GAN.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()), flush=True)

        if is_save_step(step, opt):
            save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_step_%06d.pth' % (step + 1)), opt)
            save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_step_%06d.pth' % (step + 1)), opt)
    run_training_loop(opt, step_fn, extra_steps=0)

def run(opt: Any) -> None:
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    test_loader = None
    if not opt.no_test_visualize:
        train_bsize = opt.batch_size
        opt.batch_size = opt.num_test_visualize
        opt.dataroot = opt.test_dataroot
        opt.datamode = 'test'
        opt.data_list = opt.test_data_list
        test_dataset = CPDatasetTest(opt, variant="condition")
        opt.batch_size = train_bsize
        val_dataset = Subset(test_dataset, np.arange(2000))
        test_loader = CPDataLoader(opt, test_dataset)
        val_loader = CPDataLoader(opt, val_dataset)
    board = create_summary_writer(opt)

    input1_nc = 4
    input2_nc = opt.semantic_nc + 3
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2=opt.Ddownx2, Ddropout=opt.Ddropout, n_layers_D=3, spectral=opt.spectral, num_D=opt.num_D)

    if not opt.tocg_checkpoint == '' and os.path.exists(opt.tocg_checkpoint):
        load_checkpoint(tocg, opt.tocg_checkpoint)

    train(opt, train_loader, val_loader, test_loader, board, tocg, D)

    save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_final.pth'), opt)
    save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_final.pth'), opt)
    print("Finished training %s!" % opt.name)

__all__ = [
    "train",
    "run",
]