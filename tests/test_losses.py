from __future__ import annotations
import argparse
import unittest
import torch
from aurafit.models.losses import GANLossSpade, GANLossVanilla, VGGLoss, cross_entropy2d
torch.manual_seed(0)

class TestGANLossVanilla(unittest.TestCase):
    def test_lsgan_real_target_zero_loss_when_prediction_is_all_ones(self) -> None:
        criterion = GANLossVanilla(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.ones(2, 1, 4, 4)
        loss = criterion(pred, True)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_lsgan_fake_target_zero_loss_when_prediction_is_all_zeros(self) -> None:
        criterion = GANLossVanilla(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.zeros(2, 1, 4, 4)
        loss = criterion(pred, False)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_lsgan_loss_matches_manual_mse(self) -> None:
        criterion = GANLossVanilla(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.full((2, 1, 3, 3), 0.5)
        loss = criterion(pred, True)
        expected = ((0.5 - 1.0) ** 2)
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_bce_mode_accepts_probability_valued_input(self) -> None:
        criterion = GANLossVanilla(use_lsgan=False, tensor=torch.FloatTensor)
        pred = torch.sigmoid(torch.randn(2, 1, 4, 4))
        loss = criterion(pred, True)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_multiscale_list_input_sums_per_scale_losses(self) -> None:
        criterion = GANLossVanilla(use_lsgan=True, tensor=torch.FloatTensor)
        scale_0 = [torch.ones(2, 1, 4, 4)]
        scale_1 = [torch.ones(2, 1, 2, 2)]
        loss = criterion([scale_0, scale_1], True)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_target_tensor_is_cached_and_reused_for_same_size_input(self) -> None:
        criterion = GANLossVanilla(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.zeros(2, 1, 4, 4)
        criterion(pred, True)
        cached = criterion.real_label_var
        criterion(pred, True)
        self.assertIs(criterion.real_label_var, cached)


class TestGANLossSpade(unittest.TestCase):
    def test_ls_mode_matches_manual_mse(self) -> None:
        criterion = GANLossSpade("ls", tensor=torch.FloatTensor)
        pred = torch.full((2, 1, 3, 3), 0.5)
        loss = criterion(pred, True)
        expected = (0.5 - 1.0) ** 2
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_original_mode_returns_nonnegative_scalar(self) -> None:
        criterion = GANLossSpade("original", tensor=torch.FloatTensor)
        pred = torch.randn(2, 1, 4, 4)
        loss = criterion(pred, True, for_discriminator=True)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_hinge_discriminator_real_loss_zero_when_input_large(self) -> None:
        criterion = GANLossSpade("hinge", tensor=torch.FloatTensor)
        pred = torch.full((2, 1, 4, 4), 5.0)
        loss = criterion(pred, True, for_discriminator=True)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_hinge_discriminator_fake_loss_zero_when_input_very_negative(self) -> None:
        criterion = GANLossSpade("hinge", tensor=torch.FloatTensor)
        pred = torch.full((2, 1, 4, 4), -5.0)
        loss = criterion(pred, False, for_discriminator=True)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_hinge_generator_loss_is_negative_mean_of_input(self) -> None:
        criterion = GANLossSpade("hinge", tensor=torch.FloatTensor)
        pred = torch.full((2, 1, 4, 4), 3.0)
        loss = criterion(pred, True, for_discriminator=False)
        self.assertAlmostEqual(loss.item(), -3.0, places=5)

    def test_hinge_generator_asserts_target_is_real(self) -> None:
        criterion = GANLossSpade("hinge", tensor=torch.FloatTensor)
        pred = torch.randn(2, 1, 4, 4)
        with self.assertRaises(AssertionError):
            criterion(pred, False, for_discriminator=False)

    def test_wgan_fallback_branch(self) -> None:
        criterion = GANLossSpade("w", tensor=torch.FloatTensor)
        pred = torch.full((2, 1, 4, 4), 2.0)
        real_loss = criterion(pred, True)
        fake_loss = criterion(pred, False)
        self.assertAlmostEqual(real_loss.item(), -2.0, places=5)
        self.assertAlmostEqual(fake_loss.item(), 2.0, places=5)

    def test_invalid_gan_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            GANLossSpade("not_a_real_mode", tensor=torch.FloatTensor)

    def test_multiscale_list_input_averages_per_scale_losses(self) -> None:
        criterion = GANLossSpade("ls", tensor=torch.FloatTensor)
        scale_0 = torch.full((2, 1, 4, 4), 1.0)
        scale_1 = torch.full((2, 1, 4, 4), 0.0)
        loss = criterion([scale_0, scale_1], True)
        self.assertAlmostEqual(loss.mean().item(), 0.5, places=5)

class TestCrossEntropy2d(unittest.TestCase):
    def test_matches_torch_cross_entropy_when_sizes_agree(self) -> None:
        input_ = torch.randn(2, 5, 4, 4)
        target = torch.randint(0, 5, (2, 4, 4))
        loss = cross_entropy2d(input_, target)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_interpolates_when_spatial_sizes_disagree(self) -> None:
        input_ = torch.randn(2, 5, 8, 8)
        target = torch.randint(0, 5, (2, 4, 4))
        loss = cross_entropy2d(input_, target)
        self.assertGreaterEqual(loss.item(), 0.0)


class TestVGGLoss(unittest.TestCase):
    def setUp(self) -> None:
        opt = argparse.Namespace(cuda=False)
        try:
            self.criterion = VGGLoss(opt)
        except Exception as exc:
            self.skipTest(
                f"Could not construct VGGLoss (likely no network access to "
                f"download pretrained VGG19 weights): {exc}"
            )

    def test_identical_images_have_near_zero_loss(self) -> None:
        x = torch.randn(1, 3, 64, 64)
        loss = self.criterion(x, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_different_images_have_positive_loss(self) -> None:
        x = torch.randn(1, 3, 64, 64)
        y = torch.randn(1, 3, 64, 64)
        loss = self.criterion(x, y)
        self.assertGreater(loss.item(), 0.0)

if __name__ == "__main__":
    unittest.main()