"""Loss: weighted cross-entropy + soft Dice on the foreground.

``y_true`` packs the one-hot target and the per-cell weight as the last
channel, (…, h, w, C + 1), so a plain ``model.fit`` can carry both.

``foreground_weight`` scales the loss on cells of any non-background class:
objects usually cover far fewer cells than the background, and missing one
is the error that matters.
"""

from __future__ import annotations

import tensorflow as tf


def pack(target: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
    return tf.concat([target, weight[..., None]], axis=-1)


def unpack(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return y_true[..., :-1], y_true[..., -1]


def make_loss(foreground_weight: float = 2.0, dice_weight: float = 0.5):
    def loss(y_true, y_pred):
        target, weight = unpack(y_true)
        p = tf.clip_by_value(y_pred, 1e-7, 1.0)
        fg_true = 1.0 - target[..., 0]
        cell_w = weight * (1.0 + (foreground_weight - 1.0) * fg_true)
        ce = -tf.reduce_sum(target * tf.math.log(p), axis=-1)
        ce = tf.reduce_sum(ce * cell_w) / tf.maximum(tf.reduce_sum(cell_w), 1.0)

        # Dice per foreground channel, averaged, over weighted cells. With the
        # +1 smoothing, a channel absent from the batch only costs anything
        # where the model predicts it.
        axes = tf.range(tf.rank(target) - 1)
        w = weight[..., None]
        inter = tf.reduce_sum(w * target * y_pred, axis=axes)[1:]
        denom = tf.reduce_sum(w * (target + y_pred), axis=axes)[1:]
        dice = tf.reduce_mean((2.0 * inter + 1.0) / (denom + 1.0))
        return ce + dice_weight * (1.0 - dice)

    loss.__name__ = "seg_loss"
    return loss
