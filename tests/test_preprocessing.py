from __future__ import annotations
import unittest
import numpy as np
from PIL import Image
from aurafit.preprocessing.parse_agnostic import get_im_parse_agnostic

_RAW_UPPER_1 = 5
_RAW_UPPER_2 = 6
_RAW_UPPER_3 = 7
_RAW_NECK = 10
_RAW_UNRELATED = 2

def _make_pose_data(w: int, h: int) -> np.ndarray:
    pose_data = np.zeros((18, 2), dtype=np.float64)
    for i in range(18):
        pose_data[i] = [
            w * (0.2 + 0.6 * (i / 17.0)),
            h * (0.2 + 0.6 * (i / 17.0)),
        ]
    return pose_data

def _make_raw_parse_image(w: int, h: int) -> Image.Image:
    array = np.full((h, w), _RAW_UNRELATED, dtype=np.uint8)
    array[: h // 2, : w // 2] = _RAW_UPPER_1
    array[h // 2 :, w // 2 :] = _RAW_NECK
    return Image.fromarray(array, mode="L")


class TestGetImParseAgnosticDeterminism(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 64
        self.height = 96
        self.im_parse = _make_raw_parse_image(self.width, self.height)

    def test_deterministic_given_identical_inputs(self) -> None:
        pose_data_1 = _make_pose_data(self.width, self.height)
        pose_data_2 = _make_pose_data(self.width, self.height)

        result_1 = get_im_parse_agnostic(
            self.im_parse.copy(), pose_data_1, w=self.width, h=self.height
        )
        result_2 = get_im_parse_agnostic(
            self.im_parse.copy(), pose_data_2, w=self.width, h=self.height
        )

        array_1 = np.array(result_1)
        array_2 = np.array(result_2)
        self.assertTrue(np.array_equal(array_1, array_2))

    def test_output_has_same_size_and_mode_as_input(self) -> None:
        pose_data = _make_pose_data(self.width, self.height)
        result = get_im_parse_agnostic(
            self.im_parse.copy(), pose_data, w=self.width, h=self.height
        )
        self.assertEqual(result.size, self.im_parse.size)
        self.assertEqual(result.mode, self.im_parse.mode)

    def test_upper_and_neck_regions_are_erased(self) -> None:
        pose_data = _make_pose_data(self.width, self.height)
        result_array = np.array(
            get_im_parse_agnostic(self.im_parse.copy(), pose_data, w=self.width, h=self.height)
        )
        original_array = np.array(self.im_parse)

        upper_mask = np.isin(original_array, [_RAW_UPPER_1, _RAW_UPPER_2, _RAW_UPPER_3])
        neck_mask = original_array == _RAW_NECK

        self.assertTrue(np.all(result_array[upper_mask] == 0))
        self.assertTrue(np.all(result_array[neck_mask] == 0))

    def test_unrelated_background_label_is_untouched_outside_arm_geometry(self) -> None:
        pose_data = _make_pose_data(self.width, self.height)
        result_array = np.array(
            get_im_parse_agnostic(self.im_parse.copy(), pose_data, w=self.width, h=self.height)
        )
        original_array = np.array(self.im_parse)

        corner_y, corner_x = 2, self.width - 3
        self.assertEqual(original_array[corner_y, corner_x], _RAW_UNRELATED)
        self.assertEqual(result_array[corner_y, corner_x], _RAW_UNRELATED)

    def test_does_not_mutate_input_image_object_identity_issues(self) -> None:
        pose_data = _make_pose_data(self.width, self.height)
        result = get_im_parse_agnostic(
            self.im_parse, pose_data, w=self.width, h=self.height
        )
        self.assertIsNot(result, self.im_parse)

if __name__ == "__main__":
    unittest.main()