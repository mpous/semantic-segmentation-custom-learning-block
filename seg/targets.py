"""Training targets: decoded label grid -> one-hot per output cell + loss weight.

The label grid (160 x 120 by default) is finer than the model output
(input / 8 = 40 x 40 at 320 x 320), so each output cell covers several label
cells. Per output cell, from the fraction of each class inside it:

* the foreground class with the largest share wins if that share is at least
  ``threshold`` of the cell's labeled (non-ignore) area, otherwise the
  background. A low threshold keeps small objects (a ball, a far obstacle)
  from vanishing into the background.
* a cell that is mostly ``ignore`` gets weight 0.
"""

from __future__ import annotations

import tensorflow as tf

from .classes import ClassSpec


def grid_fractions(grid_onehot: tf.Tensor, out_h: int, out_w: int) -> tf.Tensor:
    """(…, rows, cols, K) one-hot (or soft) grid -> class fractions at (out_h, out_w)."""
    return tf.image.resize(grid_onehot, (out_h, out_w), method="area")


def build_targets(frac: tf.Tensor, spec: ClassSpec, threshold: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Class fractions (…, h, w, K) -> (one-hot over output channels, weight)."""
    ig = spec.ignore_class
    ignored = frac[..., ig] if ig is not None else tf.zeros_like(frac[..., 0])
    labeled = 1.0 - ignored
    weight = tf.cast(labeled >= 0.5, tf.float32)

    fg = spec.foreground
    fg_frac = tf.gather(frac, fg, axis=-1)
    best = tf.gather(tf.constant(fg, tf.int32), tf.argmax(fg_frac, axis=-1, output_type=tf.int32))
    is_fg = tf.reduce_max(fg_frac, axis=-1) >= threshold * tf.maximum(labeled, 1e-6)
    cls = tf.where(is_fg, best, 0)
    return tf.one_hot(cls, spec.num_channels), weight
