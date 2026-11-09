"""Augmentation, applied to the image and its label grid together.

Only what a fixed, upright camera can actually see: horizontal flips,
zoom-in crops and lighting/colour/noise changes. No vertical flips and no
rotation, since the ground is always at the bottom of the frame.
"""

from __future__ import annotations

import tensorflow as tf


def augment(image: tf.Tensor, grid: tf.Tensor, min_scale: float = 0.75) -> tuple[tf.Tensor, tf.Tensor]:
    """image (H, W, C) in [0, 1]; grid (rows, cols, K) class one-hot. Returns both augmented."""
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    rows, cols = tf.shape(grid)[0], tf.shape(grid)[1]

    if tf.random.uniform(()) < 0.5:
        image = tf.image.flip_left_right(image)
        grid = tf.image.flip_left_right(grid)

    # Zoom: crop a random window of the same aspect ratio and scale it back up.
    s = tf.random.uniform((), min_scale, 1.0)
    y0 = tf.random.uniform((), 0.0, 1.0 - s)
    x0 = tf.random.uniform((), 0.0, 1.0 - s)
    box = tf.reshape(tf.stack([y0, x0, y0 + s, x0 + s]), (1, 4))
    image = tf.image.crop_and_resize(image[None], box, [0], tf.stack([h, w]))[0]
    # Bilinear on one-hot gives soft class fractions, which build_targets expects.
    grid = tf.image.crop_and_resize(grid[None], box, [0], tf.stack([rows, cols]))[0]

    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    if image.shape[-1] == 3:
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_hue(image, 0.03)
    noise = tf.random.uniform((), 0.0, 0.03)
    image = image + tf.random.normal(tf.shape(image), stddev=noise)
    return tf.clip_by_value(image, 0.0, 1.0), grid
