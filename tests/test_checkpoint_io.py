from __future__ import annotations
import argparse
import os
import tempfile
import unittest
from collections import OrderedDict
import torch
import torch.nn as nn
from aurafit.checkpoint.io import load_checkpoint, load_checkpoint_G, save_checkpoint

def _make_opt(cuda: bool = False) -> argparse.Namespace:
    return argparse.Namespace(cuda=cuda)


class _TinyModel(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)


class _TinyAliasModel(nn.Module):
    def __init__(self, features: int = 4) -> None:
        super().__init__()
        self.alias_layer = nn.Linear(features, features)


class TestSaveAndLoadCheckpointRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_round_trip_preserves_parameters(self) -> None:
        opt = _make_opt(cuda=False)
        source_model = _TinyModel()
        checkpoint_path = os.path.join(self._tmpdir.name, "nested", "dir", "model.pth")

        save_checkpoint(source_model, checkpoint_path, opt)
        self.assertTrue(os.path.exists(checkpoint_path))

        target_model = _TinyModel()
        self.assertFalse(
            torch.equal(source_model.linear.weight, target_model.linear.weight)
        )

        load_checkpoint(target_model, checkpoint_path, opt)

        self.assertTrue(
            torch.equal(source_model.linear.weight, target_model.linear.weight)
        )
        self.assertTrue(
            torch.equal(source_model.linear.bias, target_model.linear.bias)
        )

    def test_save_checkpoint_creates_missing_directories(self) -> None:
        opt = _make_opt(cuda=False)
        model = _TinyModel()
        checkpoint_path = os.path.join(
            self._tmpdir.name, "does", "not", "exist", "yet", "ckpt.pth"
        )
        self.assertFalse(os.path.exists(os.path.dirname(checkpoint_path)))
        save_checkpoint(model, checkpoint_path, opt)
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_load_checkpoint_is_non_strict(self) -> None:
        opt = _make_opt(cuda=False)
        source_model = _TinyModel()
        checkpoint_path = os.path.join(self._tmpdir.name, "partial.pth")
        save_checkpoint(source_model, checkpoint_path, opt)
        class _MismatchedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.extra = nn.Linear(2, 2)
        target_model = _MismatchedModel()
        load_checkpoint(target_model, checkpoint_path, opt)
        self.assertTrue(
            torch.equal(source_model.linear.weight, target_model.linear.weight)
        )

    def test_load_checkpoint_raises_on_missing_path(self) -> None:
        opt = _make_opt(cuda=False)
        model = _TinyModel()
        missing_path = os.path.join(self._tmpdir.name, "nonexistent.pth")

        with self.assertRaises(RuntimeError):
            load_checkpoint(model, missing_path, opt)


class TestLoadCheckpointGKeyRemapping(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_missing_path_is_a_silent_noop(self) -> None:
        opt = _make_opt(cuda=False)
        model = _TinyAliasModel()
        original_weight = model.alias_layer.weight.clone()
        missing_path = os.path.join(self._tmpdir.name, "nonexistent.pth")
        load_checkpoint_G(model, missing_path, opt)
        self.assertTrue(torch.equal(model.alias_layer.weight, original_weight))

    def test_legacy_ace_and_spade_keys_are_remapped_and_loaded_strictly(self) -> None:
        opt = _make_opt(cuda=False)
        source_model = _TinyAliasModel()

        legacy_state_dict = OrderedDict()
        for key, value in source_model.state_dict().items():
            legacy_key = key.replace("alias", "ace")
            legacy_key = legacy_key.replace("ace_layer", "ace_layer.Spade")
            legacy_state_dict[legacy_key] = value.clone()
        legacy_state_dict._metadata = OrderedDict(
            (k.replace("alias", "ace").replace("ace_layer", "ace_layer.Spade"), v)
            for k, v in source_model.state_dict()._metadata.items()
        )

        checkpoint_path = os.path.join(self._tmpdir.name, "legacy_gen.pth")
        torch.save(legacy_state_dict, checkpoint_path)

        target_model = _TinyAliasModel()
        self.assertFalse(
            torch.equal(source_model.alias_layer.weight, target_model.alias_layer.weight)
        )

        load_checkpoint_G(target_model, checkpoint_path, opt)

        self.assertTrue(
            torch.equal(source_model.alias_layer.weight, target_model.alias_layer.weight)
        )
        self.assertTrue(
            torch.equal(source_model.alias_layer.bias, target_model.alias_layer.bias)
        )

    def test_strict_loading_raises_on_genuine_key_mismatch(self) -> None:
        opt = _make_opt(cuda=False)
        class _DifferentShapeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.completely_different_name = nn.Linear(4, 4)

        source_model = _TinyAliasModel()
        checkpoint_path = os.path.join(self._tmpdir.name, "mismatched.pth")
        torch.save(source_model.state_dict(), checkpoint_path)

        target_model = _DifferentShapeModel()
        with self.assertRaises(RuntimeError):
            load_checkpoint_G(target_model, checkpoint_path, opt)

if __name__ == "__main__":
    unittest.main()