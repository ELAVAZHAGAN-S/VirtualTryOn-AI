from __future__ import annotations
import argparse
import unittest
import torch
from aurafit.models.condition_generator import ConditionGenerator
from aurafit.models.discriminators import (
    MultiscaleDiscriminatorSpade,
    MultiscaleDiscriminatorVanilla,
    NLayerDiscriminatorSpade,
    NLayerDiscriminatorVanilla,
    define_D,
)
from aurafit.models.init import get_norm_layer
from aurafit.models.spade.generator import SPADEGenerator
torch.manual_seed(0)

def _condition_opt(cuda: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        warp_feature="T1",
        out_layer="relu",
        cuda=cuda,
        upsample="bilinear",
    )

def _spade_opt(
    fine_height: int = 128,
    fine_width: int = 128,
    ngf: int = 8,
    gen_semantic_nc: int = 7,
    norm_G: str = "spectralaliasinstance",
    norm_D: str = "spectralinstance",
    ndf: int = 8,
    n_layers_D: int = 3,
    num_D: int = 2,
    no_ganFeat_loss: bool = True,
    num_upsampling_layers: str = "normal",
    cuda: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        fine_height=fine_height,
        fine_width=fine_width,
        ngf=ngf,
        gen_semantic_nc=gen_semantic_nc,
        norm_G=norm_G,
        norm_D=norm_D,
        ndf=ndf,
        n_layers_D=n_layers_D,
        num_D=num_D,
        no_ganFeat_loss=no_ganFeat_loss,
        num_upsampling_layers=num_upsampling_layers,
        cuda=cuda,
    )

class TestConditionGeneratorForwardShapes(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = _condition_opt()
        self.batch_size = 2
        self.height = 128
        self.width = 128
        self.input1_nc = 4  # cloth (3) + cloth-mask (1)
        self.semantic_nc = 13
        self.input2_nc = self.semantic_nc + 3  # parse_agnostic + densepose
        self.output_nc = self.semantic_nc
        self.ngf = 8
        self.model = ConditionGenerator(
            self.opt,
            input1_nc=self.input1_nc,
            input2_nc=self.input2_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
        )
        self.model.eval()

    def test_forward_output_shapes(self) -> None:
        input1 = torch.randn(self.batch_size, self.input1_nc, self.height, self.width)
        input2 = torch.randn(self.batch_size, self.input2_nc, self.height, self.width)
        with torch.no_grad():
            flow_list, fake_segmap, warped_c, warped_cm = self.model(self.opt, input1, input2)

        self.assertEqual(
            fake_segmap.shape,
            (self.batch_size, self.output_nc, self.height, self.width),
        )
        self.assertEqual(
            warped_c.shape,
            (self.batch_size, self.input1_nc - 1, self.height, self.width),
        )
        self.assertEqual(
            warped_cm.shape,
            (self.batch_size, 1, self.height, self.width),
        )
        self.assertEqual(len(flow_list), 5)
        for flow in flow_list:
            self.assertEqual(flow.shape[0], self.batch_size)
            self.assertEqual(flow.shape[-1], 2)  # (x, y) displacement channels

    def test_forward_with_encoder_warp_feature(self) -> None:
        opt = _condition_opt()
        opt.warp_feature = "encoder"
        model = ConditionGenerator(
            opt,
            input1_nc=self.input1_nc,
            input2_nc=self.input2_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
        )
        model.eval()

        input1 = torch.randn(self.batch_size, self.input1_nc, self.height, self.width)
        input2 = torch.randn(self.batch_size, self.input2_nc, self.height, self.width)

        with torch.no_grad():
            _, fake_segmap, _, _ = model(opt, input1, input2)

        self.assertEqual(
            fake_segmap.shape,
            (self.batch_size, self.output_nc, self.height, self.width),
        )

    def test_forward_with_conv_out_layer(self) -> None:
        opt = _condition_opt()
        opt.out_layer = "conv"
        model = ConditionGenerator(
            opt,
            input1_nc=self.input1_nc,
            input2_nc=self.input2_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
        )
        model.eval()

        input1 = torch.randn(self.batch_size, self.input1_nc, self.height, self.width)
        input2 = torch.randn(self.batch_size, self.input2_nc, self.height, self.width)

        with torch.no_grad():
            _, fake_segmap, _, _ = model(opt, input1, input2)

        self.assertEqual(
            fake_segmap.shape,
            (self.batch_size, self.output_nc, self.height, self.width),
        )


class TestSPADEGeneratorForwardShapes(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = _spade_opt()
        self.batch_size = 2
        self.input_nc = 9
        self.model = SPADEGenerator(self.opt, self.input_nc)
        self.model.eval()

    def test_forward_output_shape(self) -> None:
        x = torch.randn(
            self.batch_size, self.input_nc, self.opt.fine_height, self.opt.fine_width
        )
        seg = torch.randn(
            self.batch_size,
            self.opt.gen_semantic_nc,
            self.opt.fine_height,
            self.opt.fine_width,
        )

        with torch.no_grad():
            output = self.model(x, seg)

        self.assertEqual(
            output.shape,
            (self.batch_size, 3, self.opt.fine_height, self.opt.fine_width),
        )
        self.assertGreaterEqual(output.min().item(), -1.0 - 1e-4)
        self.assertLessEqual(output.max().item(), 1.0 + 1e-4)

    def test_forward_with_most_upsampling_layers(self) -> None:
        opt = _spade_opt(num_upsampling_layers="most")
        model = SPADEGenerator(opt, self.input_nc)
        model.eval()
        x = torch.randn(self.batch_size, self.input_nc, opt.fine_height, opt.fine_width)
        seg = torch.randn(self.batch_size, opt.gen_semantic_nc, opt.fine_height, opt.fine_width)
        with torch.no_grad():
            output = model(x, seg)
        self.assertEqual(
            output.shape, (self.batch_size, 3, opt.fine_height, opt.fine_width)
        )

class TestVanillaDiscriminatorForwardShapes(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.input_nc = 10
        self.height = 64
        self.width = 64
        self.norm_layer = get_norm_layer("instance")

    def test_nlayer_discriminator_without_intermediate_features(self) -> None:
        model = NLayerDiscriminatorVanilla(
            self.input_nc, ndf=8, n_layers=3, norm_layer=self.norm_layer, getIntermFeat=False
        )
        model.eval()
        x = torch.randn(self.batch_size, self.input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], 1)

    def test_nlayer_discriminator_with_intermediate_features(self) -> None:
        model = NLayerDiscriminatorVanilla(
            self.input_nc, ndf=8, n_layers=3, norm_layer=self.norm_layer, getIntermFeat=True
        )
        model.eval()
        x = torch.randn(self.batch_size, self.input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), model.n_layers + 2)
        for feat in out:
            self.assertEqual(feat.shape[0], self.batch_size)

    def test_multiscale_discriminator_output_structure(self) -> None:
        num_D = 3
        model = MultiscaleDiscriminatorVanilla(
            self.input_nc,
            ndf=8,
            n_layers=3,
            norm_layer=self.norm_layer,
            num_D=num_D,
            getIntermFeat=False,
        )
        model.eval()
        x = torch.randn(self.batch_size, self.input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(len(out), num_D)
        for scale_result in out:
            self.assertIsInstance(scale_result, list)
            self.assertEqual(len(scale_result), 1)
            self.assertEqual(scale_result[0].shape[0], self.batch_size)

    def test_define_d_factory_produces_working_discriminator(self) -> None:
        num_D = 2
        model = define_D(
            input_nc=self.input_nc,
            ndf=8,
            n_layers_D=3,
            num_D=num_D,
            gpu_ids=[],
        )
        model.eval()
        x = torch.randn(self.batch_size, self.input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(len(out), num_D)
        for scale_result in out:
            self.assertEqual(scale_result[0].shape[0], self.batch_size)


class TestSpadeDiscriminatorForwardShapes(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.height = 64
        self.width = 64

    def test_nlayer_discriminator_spade_no_ganfeat_loss(self) -> None:
        opt = _spade_opt(no_ganFeat_loss=True)
        input_nc = opt.gen_semantic_nc + 3
        model = NLayerDiscriminatorSpade(opt)
        model.eval()
        x = torch.randn(self.batch_size, input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], 1)

    def test_nlayer_discriminator_spade_with_ganfeat_loss(self) -> None:
        opt = _spade_opt(no_ganFeat_loss=False)
        input_nc = opt.gen_semantic_nc + 3
        model = NLayerDiscriminatorSpade(opt)
        model.eval()
        x = torch.randn(self.batch_size, input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, list)
        for feat in out:
            self.assertEqual(feat.shape[0], self.batch_size)

    def test_multiscale_discriminator_spade_output_structure(self) -> None:
        opt = _spade_opt(no_ganFeat_loss=True, num_D=3)
        input_nc = opt.gen_semantic_nc + 3
        model = MultiscaleDiscriminatorSpade(opt)
        model.eval()
        x = torch.randn(self.batch_size, input_nc, self.height, self.width)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(len(out), opt.num_D)
        for scale_result in out:
            self.assertIsInstance(scale_result, list)
            self.assertEqual(len(scale_result), 1)
            self.assertEqual(scale_result[0].shape[0], self.batch_size)

if __name__ == "__main__":
    unittest.main()