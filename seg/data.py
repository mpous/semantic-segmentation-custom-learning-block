"""Loading what Studio hands the block, and turning it into training data.

``--data-directory`` holds:

* ``X_split_train.npy`` / ``X_split_test.npy``: images, NHWC, scaled per
  ``imageInputScaling`` (this block declares ``0..1``). "test" is the
  validation split Studio made from the training set.
* ``Y_split_train.npy`` / ``Y_split_test.npy``: for object detection these are
  JSON text despite the extension:
  ``[{"sampleId": 1, "boundingBoxes": [{"label": 1, "x": 0, "y": 0, "w": 8, "h": 4}]}]``
  with 1-based labels into ``classes`` and boxes in pixels of the resized
  impulse input.
* ``sample_id_details.json``: sample ids per split (not needed here; Y has them).

``--info-file`` (``train_input.json``, not present in local runs) gives the
label list in ``classes`` and the input shape in ``inputShapeString``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from .augment import augment
from .classes import ClassSpec
from .losses import pack
from .runboxes import Box, decode_boxes
from .targets import build_targets, grid_fractions


@dataclass
class Split:
    #: (N, H, W, C) uint8: the 0..1 input stored at 1/255 steps to save memory.
    images: np.ndarray
    #: (N, rows, cols) uint8 labeler class ids.
    grids: np.ndarray
    sample_ids: list[int]


def read_info(path: str | None) -> dict:
    if not path or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def input_shape_from_info(info: dict) -> tuple[int, ...] | None:
    s = info.get("inputShapeString")
    if not s:
        return None
    return tuple(int(v) for v in re.findall(r"\d+", s))


def read_boxes(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        y = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SystemExit(
            f"{path.name} is not bounding-box JSON. This block needs an object detection project "
            '(Dashboard → Project info → Labeling method: "Bounding boxes").') from None
    if not isinstance(y, list):
        raise SystemExit(f"{path.name}: expected a JSON list of samples")
    return y


def load_images(path: Path, shape: tuple[int, ...] | None) -> np.ndarray:
    x = np.load(path, mmap_mode="r")
    if x.ndim == 2 and shape:
        x = x.reshape((x.shape[0], *shape))
    if x.ndim != 4:
        raise SystemExit(f"{path.name}: expected images (N, H, W, C), got shape {x.shape}. "
                         "Use an Image processing block in the impulse.")
    if x.size:
        lo, hi = float(np.min(x)), float(np.max(x))
        if lo < -0.01 or hi > 1.01:
            raise SystemExit(f"{path.name}: pixel values are in {lo:.2f}..{hi:.2f}, expected 0..1. "
                             'Set "Image input scaling" to 0..1 in the learning block settings.')
    return np.round(np.asarray(x, np.float32) * 255.0).astype(np.uint8)


def resolve_labels(info: dict, override: Sequence[str], spec_names: Sequence[str],
                   default_class: str, max_label: int) -> list[str]:
    """Studio's label list: train_input.json, else --studio-labels, else a guess (local runs)."""
    if info.get("classes"):
        return list(info["classes"])
    if override:
        return list(override)
    # Studio sorts labels alphabetically; the labeler's non-default classes,
    # cut to the highest label index in the data, is the best local guess.
    guess = sorted(n for n in spec_names if n != default_class)[:max(max_label, 1)]
    print(f"No train_input.json (local run): assuming Studio labels {guess}. "
          "Pass --studio-labels to override.", flush=True)
    return guess


def max_label(*ys: list[dict]) -> int:
    return max((b["label"] for y in ys for s in y for b in s.get("boundingBoxes", [])), default=0)


def decode_split(x: np.ndarray, y: list[dict], spec: ClassSpec, cols: int, rows: int) -> Split:
    if len(y) != len(x):
        raise SystemExit(f"{len(x)} images but {len(y)} label entries")
    h, w = x.shape[1:3]
    grids = np.empty((len(y), rows, cols), np.uint8)
    unknown: set[str] = set()
    for i, s in enumerate(y):
        boxes = []
        for b in s.get("boundingBoxes", []):
            idx = int(b["label"]) - 1
            if not 0 <= idx < len(spec.studio_labels):
                raise SystemExit(f"Sample {s.get('sampleId')}: label index {b['label']} is outside "
                                 f"the label list {list(spec.studio_labels)}")
            boxes.append(Box(spec.studio_labels[idx], b["x"], b["y"], b["w"], b["h"]))
        grids[i], unk = decode_boxes(boxes, spec.names, spec.default_class, cols, rows, w, h)
        unknown |= unk
    if unknown:
        print(f"Warning: boxes with labels {sorted(unknown)} were ignored (not labeler classes)", flush=True)
    return Split(x, grids, [int(s.get("sampleId", -1)) for s in y])


def make_dataset(split: Split, spec: ClassSpec, out_hw: tuple[int, int], threshold: float,
                 batch_size: int, training: bool) -> tf.data.Dataset:
    k = len(spec.names)
    ds = tf.data.Dataset.from_tensor_slices((split.images, split.grids))
    if training:
        ds = ds.shuffle(len(split.images), reshuffle_each_iteration=True)

    def prep(image, grid):
        image = tf.cast(image, tf.float32) / 255.0
        onehot = tf.one_hot(tf.cast(grid, tf.int32), k)
        if training:
            image, onehot = augment(image, onehot)
        target, weight = build_targets(grid_fractions(onehot, *out_hw), spec, threshold)
        return image, pack(target, weight)

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
