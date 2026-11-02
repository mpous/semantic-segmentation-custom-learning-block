"""The decoder must give exactly what the extension's decoder gives.

tests/fixtures/runboxes.json is written by the labeler extension
(``npm run fixtures`` in semantic-segmentation-labeler-ei-extension).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from seg.runboxes import Box, cell_span, decode_boxes

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "runboxes.json").read_text())
CASES = [(i, name) for i, c in enumerate(FIXTURES["cases"]) for name in c["frames"]]


def as_grid(s: str, rows: int, cols: int) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8).reshape(rows, cols) - ord("0")


@pytest.mark.parametrize("case,frame", CASES)
def test_matches_extension(case, frame):
    c = FIXTURES["cases"][case]
    f = c["frames"][frame]
    boxes = [Box(b["label"], b["x"], b["y"], b["width"], b["height"]) for b in f["boxes"]]
    grid, unknown = decode_boxes(boxes, c["classes"], c["defaultClass"], c["cols"], c["rows"], f["width"], f["height"])
    assert not unknown
    np.testing.assert_array_equal(grid, as_grid(f["grid"], c["rows"], c["cols"]))


@pytest.mark.parametrize("case", range(len(FIXTURES["cases"])))
def test_round_trip_in_every_unedited_frame(case):
    c = FIXTURES["cases"][case]
    for name in ("image", "impulse", "normalised"):
        assert c["frames"][name]["grid"] == c["encoded"], name


def test_unknown_labels_are_skipped():
    grid, unknown = decode_boxes([Box("chair", 0, 0, 8, 8), Box("obstacle", 4, 0, 4, 4)],
                                 ["floor", "obstacle"], 0, 4, 4, 16, 16)
    assert unknown == {"chair"}
    assert grid.tolist() == [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_cell_span():
    assert [cell_span(i, 3, 10) for i in range(3)] == [(0, 3), (3, 6), (6, 10)]
