from __future__ import annotations
import unittest
from aurafit.constants.labels import (
    BACKGROUND,
    BOTTOM,
    FACE,
    HAIR,
    LEFT_ARM,
    LEFT_LEG,
    LEFT_SHOE,
    NOISE,
    RAW_PARSE_NUM_CLASSES,
    RIGHT_ARM,
    RIGHT_LEG,
    RIGHT_SHOE,
    SEMANTIC_LABELS,
    SEMANTIC_NUM_CLASSES,
    SOCKS,
    UPPER,
    get_raw_parse_indices,
    get_semantic_class_name,
    iter_semantic_label_items,
)

_ORIGINAL_LABELS = {
    0: ['background', [0, 10]],
    1: ['hair', [1, 2]],
    2: ['face', [4, 13]],
    3: ['upper', [5, 6, 7]],
    4: ['bottom', [9, 12]],
    5: ['left_arm', [14]],
    6: ['right_arm', [15]],
    7: ['left_leg', [16]],
    8: ['right_leg', [17]],
    9: ['left_shoe', [18]],
    10: ['right_shoe', [19]],
    11: ['socks', [8]],
    12: ['noise', [3, 11]],
}

class TestSemanticLabelConstants(unittest.TestCase):
    def test_class_counts(self) -> None:
        self.assertEqual(SEMANTIC_NUM_CLASSES, 13)
        self.assertEqual(RAW_PARSE_NUM_CLASSES, 20)
        self.assertEqual(len(SEMANTIC_LABELS), SEMANTIC_NUM_CLASSES)

    def test_matches_original_literal_dict_exactly(self) -> None:
        self.assertEqual(list(SEMANTIC_LABELS.keys()), list(_ORIGINAL_LABELS.keys()))
        for key in _ORIGINAL_LABELS:
            self.assertEqual(SEMANTIC_LABELS[key][0], _ORIGINAL_LABELS[key][0])
            self.assertEqual(SEMANTIC_LABELS[key][1], _ORIGINAL_LABELS[key][1])

    def test_named_index_constants_match_dict_keys(self) -> None:
        named_constants = {
            BACKGROUND: 'background',
            HAIR: 'hair',
            FACE: 'face',
            UPPER: 'upper',
            BOTTOM: 'bottom',
            LEFT_ARM: 'left_arm',
            RIGHT_ARM: 'right_arm',
            LEFT_LEG: 'left_leg',
            RIGHT_LEG: 'right_leg',
            LEFT_SHOE: 'left_shoe',
            RIGHT_SHOE: 'right_shoe',
            SOCKS: 'socks',
            NOISE: 'noise',
        }
        for index, expected_name in named_constants.items():
            self.assertEqual(SEMANTIC_LABELS[index][0], expected_name)

    def test_raw_indices_form_a_partition_of_0_to_19(self) -> None:
        all_raw_indices = []
        for _, _, raw_indices in iter_semantic_label_items():
            all_raw_indices.extend(raw_indices)
        self.assertEqual(len(all_raw_indices), RAW_PARSE_NUM_CLASSES)
        self.assertEqual(sorted(all_raw_indices), list(range(RAW_PARSE_NUM_CLASSES)))

    def test_get_semantic_class_name(self) -> None:
        self.assertEqual(get_semantic_class_name(UPPER), 'upper')
        self.assertEqual(get_semantic_class_name(BACKGROUND), 'background')
        self.assertEqual(get_semantic_class_name(NOISE), 'noise')

    def test_get_raw_parse_indices(self) -> None:
        self.assertEqual(get_raw_parse_indices(FACE), [4, 13])
        self.assertEqual(get_raw_parse_indices(UPPER), [5, 6, 7])
        self.assertEqual(get_raw_parse_indices(SOCKS), [8])

    def test_iter_semantic_label_items_order_and_content(self) -> None:
        items = iter_semantic_label_items()
        self.assertEqual(len(items), SEMANTIC_NUM_CLASSES)
        for (index, name, raw_indices), (orig_index, (orig_name, orig_raw)) in zip(
            items, _ORIGINAL_LABELS.items()
        ):
            self.assertEqual(index, orig_index)
            self.assertEqual(name, orig_name)
            self.assertEqual(raw_indices, orig_raw)

    def test_label_ops_indices_match_semantic_scheme(self) -> None:
        self.assertEqual(UPPER, 3)
        self.assertEqual(LEFT_ARM, 5)
        self.assertEqual(RIGHT_ARM, 6)

if __name__ == '__main__':
    unittest.main()