"""Run-box decoding: the inverse of the labeler extension's encoder.

The Segmentation Labeler extension stores each mask as bounding boxes because
a learning block receives nothing else about a sample (no metadata, no
network). Its contract, from the extension README ("Data format"):

* The image is split into a ``cols`` x ``rows`` grid (default 160 x 120).
* Every cell starts as the default class (``floor``).
* Each box with a known label sets every cell whose *centre* lies inside the
  box to the box's class. Later boxes win where boxes overlap.

The maths matches ``decodeBoxes`` in ``src/masks/encode.ts`` operation for
operation (same float64 arithmetic), so both sides produce identical grids.
``tests/test_runboxes.py`` checks this against fixtures written by the
extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Box:
    label: str
    x: float
    y: float
    width: float
    height: float


def cell_span(i: int, cells: int, size: int) -> tuple[int, int]:
    """Pixel span [start, end) of cell ``i`` when ``size`` pixels are split into ``cells``."""
    return math.floor(i * size / cells), math.floor((i + 1) * size / cells)


def decode_boxes(
    boxes: Iterable[Box],
    class_names: Sequence[str],
    default_class: int,
    cols: int,
    rows: int,
    width: float,
    height: float,
) -> tuple[np.ndarray, set[str]]:
    """Boxes -> class grid of shape (rows, cols), uint8.

    ``width``/``height`` are the size of the frame the boxes are expressed in:
    image pixels, the resized impulse input, or 1.0 for normalised boxes.
    Returns the grid and the set of box labels not in ``class_names``
    (those boxes are skipped, as in the extension).
    """
    index: Mapping[str, int] = {name: k for k, name in enumerate(class_names)}
    grid = np.full((rows, cols), default_class, dtype=np.uint8)
    unknown: set[str] = set()
    cell_w = width / cols
    cell_h = height / rows

    for b in boxes:
        k = index.get(b.label)
        if k is None:
            unknown.add(b.label)
            continue
        # Cells whose centre (i + 0.5) * cell lies in [x, x + width).
        c0 = max(0, math.ceil(b.x / cell_w - 0.5))
        c1 = min(cols - 1, math.ceil((b.x + b.width) / cell_w - 0.5) - 1)
        r0 = max(0, math.ceil(b.y / cell_h - 0.5))
        r1 = min(rows - 1, math.ceil((b.y + b.height) / cell_h - 0.5) - 1)
        if c1 >= c0 and r1 >= r0:
            grid[r0:r1 + 1, c0:c1 + 1] = k
    return grid, unknown
