"""Training targets: decoded label grid -> one-hot per output cell + loss weight.

The label grid (160 x 120 by default) is finer than the model output
(input / 8 = 40 x 30 at 320 x 240), so each output cell covers several label
cells. Per output cell, from the fraction of each class inside it:

* the foreground class with the largest share wins if that share is at least
  ``threshold`` of the cell's labeled (non-ignore) area, otherwise the default
  class (``floor``). A low threshold leans to "obstacle": the safe side.
* in ``exclude`` mode a cell that is mostly ``ignore`` gets weight 0.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .classes import ClassSpec


def grid_fractions(grid_onehot: tf.Tensor, out_h: int, out_w: int) -> tf.Tensor:
    """(…, rows, cols, K) one-hot (or soft) grid -> class fractions at (out_h, out_w)."""
    return tf.image.resize(grid_onehot, (out_h, out_w), method="area")


def build_targets(frac: tf.Tensor, spec: ClassSpec, threshold: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Class fractions (…, h, w, K) -> (one-hot over output channels, weight)."""
    k_all = len(spec.names)
    ig = spec.ignore_class

    merge = np.eye(k_all, dtype=np.float32)
    if ig is not None and spec.ignore_mode == "obstacle":
        merge[ig, ig] = 0.0
        merge[ig, spec.obstacle_class] = 1.0
    frac = tf.einsum("...k,kj->...j", frac, tf.constant(merge))

    if ig is not None:
        ignored = frac[..., ig]
    else:
        ignored = tf.zeros_like(frac[..., 0])
    labeled = 1.0 - ignored
    weight = tf.cast(labeled >= 0.5, tf.float32)

    candidates = [k for k in range(k_all) if k != spec.default_class and k != ig]
    # `cls` below is never the ignore class, so its entry is a placeholder.
    channel_of = tf.constant([0 if k == ig else spec.channel_of(k) for k in range(k_all)], tf.int32)
    if candidates:
        cand_frac = tf.gather(frac, candidates, axis=-1)
        best = tf.gather(tf.constant(candidates, tf.int32), tf.argmax(cand_frac, axis=-1, output_type=tf.int32))
        is_fg = tf.reduce_max(cand_frac, axis=-1) >= threshold * tf.maximum(labeled, 1e-6)
        cls = tf.where(is_fg, best, spec.default_class)
    else:
        cls = tf.fill(tf.shape(weight), spec.default_class)
    target = tf.one_hot(tf.gather(channel_of, cls), spec.num_channels)
    return target, weight
