from __future__ import annotations
from typing import Dict, List, Tuple

RAW_PARSE_NUM_CLASSES: int = 20

BACKGROUND: int = 0
HAIR: int = 1
FACE: int = 2
UPPER: int = 3
BOTTOM: int = 4
LEFT_ARM: int = 5
RIGHT_ARM: int = 6
LEFT_LEG: int = 7
RIGHT_LEG: int = 8
LEFT_SHOE: int = 9
RIGHT_SHOE: int = 10
SOCKS: int = 11
NOISE: int = 12

SEMANTIC_NUM_CLASSES: int = 13

SEMANTIC_LABELS: Dict[int, List[object]] = {
    BACKGROUND: ['background', [0, 10]],
    HAIR:       ['hair',       [1, 2]],
    FACE:       ['face',       [4, 13]],
    UPPER:      ['upper',      [5, 6, 7]],
    BOTTOM:     ['bottom',     [9, 12]],
    LEFT_ARM:   ['left_arm',   [14]],
    RIGHT_ARM:  ['right_arm',  [15]],
    LEFT_LEG:   ['left_leg',   [16]],
    RIGHT_LEG:  ['right_leg',  [17]],
    LEFT_SHOE:  ['left_shoe',  [18]],
    RIGHT_SHOE: ['right_shoe', [19]],
    SOCKS:      ['socks',      [8]],
    NOISE:      ['noise',      [3, 11]],
}

def get_semantic_class_name(semantic_index: int) -> str:
    return str(SEMANTIC_LABELS[semantic_index][0])


def get_raw_parse_indices(semantic_index: int) -> List[int]:
    return list(SEMANTIC_LABELS[semantic_index][1])


def iter_semantic_label_items() -> List[Tuple[int, str, List[int]]]:
    return [
        (semantic_index, str(name), list(raw_indices))
        for semantic_index, (name, raw_indices) in SEMANTIC_LABELS.items()
    ]

__all__ = [
    "RAW_PARSE_NUM_CLASSES",
    "SEMANTIC_NUM_CLASSES",
    "BACKGROUND",
    "HAIR",
    "FACE",
    "UPPER",
    "BOTTOM",
    "LEFT_ARM",
    "RIGHT_ARM",
    "LEFT_LEG",
    "RIGHT_LEG",
    "LEFT_SHOE",
    "RIGHT_SHOE",
    "SOCKS",
    "NOISE",
    "SEMANTIC_LABELS",
    "get_semantic_class_name",
    "get_raw_parse_indices",
    "iter_semantic_label_items",
]