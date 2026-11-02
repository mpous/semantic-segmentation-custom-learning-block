"""The network: MobileNetV2 encoder + a light LR-ASPP decoder.

Input is the impulse's image, values in [0, 1] (rescaled to MobileNetV2's
[-1, 1] inside the model so the deployed model takes what Edge Impulse feeds
it). Output is a softmax over ``num_channels`` at 1/8 of the input size (FOMO
layout: channel 0 = background) or 1/4 with ``output_stride=4``.

Only the encoder up to 1/16 is kept: the decoder takes its context from there,
which keeps the model small enough for the UNO Q's CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

# MobileNetV2 layers at 1/4, 1/8 and 1/16 of the input size.
TAP = {4: "block_3_expand_relu", 8: "block_6_expand_relu", 16: "block_13_expand_relu"}
ALPHAS = (0.35, 0.5, 0.75, 1.0)
BN_MOMENTUM = 0.9
WEIGHTS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"


def weights_name(alpha: float) -> str:
    return f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{float(alpha)}_224_no_top.h5"


def weights_file(alpha: float) -> str:
    """ImageNet weights: from $SEG_WEIGHTS_DIR (baked into the Docker image,
    since Studio runs blocks without network), else downloaded to ~/.keras."""
    name = weights_name(alpha)
    local = os.environ.get("SEG_WEIGHTS_DIR")
    if local:
        path = Path(local) / name
        if not path.is_file():
            raise SystemExit(f"Pretrained weights {path} are missing from the image; rebuild it.")
        return str(path)
    return tf.keras.utils.get_file(name, WEIGHTS_URL + name, cache_subdir="models")


def _conv_bn_relu(x, filters, name):
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(6.0, name=f"{name}_relu")(x)


def build_model(
    height: int,
    width: int,
    channels: int,
    num_outputs: int,
    alpha: float = 0.35,
    output_stride: int = 8,
    decoder_filters: int = 64,
    pretrained: bool = True,
) -> tf.keras.Model:
    if alpha not in ALPHAS:
        raise ValueError(f"backbone width must be one of {ALPHAS}, not {alpha}")
    if output_stride not in (4, 8):
        raise ValueError(f"output stride must be 4 or 8, not {output_stride}")
    if height % 16 or width % 16:
        raise ValueError(f"Input {width}x{height} must be a multiple of 16 on both sides "
                         "(e.g. 320x240 or 160x128); change the image size in the impulse.")
    if channels not in (1, 3):
        raise ValueError(f"Expected an RGB or grayscale image input, got {channels} channels")

    inputs = layers.Input((height, width, channels), name="image")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)
    if channels == 1:
        # Grayscale impulse: repeat the channel so the ImageNet weights still apply.
        x = layers.Concatenate(name="gray_to_rgb")([x, x, x])

    backbone = tf.keras.applications.MobileNetV2(input_tensor=x, alpha=alpha, include_top=False, weights=None)
    if pretrained:
        # Same layers at any input size, so the 224 px ImageNet file always fits.
        backbone.load_weights(weights_file(alpha))
    # Keras ships these with momentum 0.999: on a few hundred images the
    # inference statistics would stay ImageNet's for most of the run.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.momentum = BN_MOMENTUM
    taps = {s: backbone.get_layer(n).output for s, n in TAP.items()}

    # LR-ASPP: 1/16 features, gated by their global average (a cheap
    # squeeze-excite), upsampled to 1/8 and fused with the 1/8 features.
    high = _conv_bn_relu(taps[16], decoder_filters, "aspp")
    gate = layers.GlobalAveragePooling2D(keepdims=True, name="aspp_pool")(taps[16])
    gate = layers.Conv2D(decoder_filters, 1, activation="sigmoid", name="aspp_gate")(gate)
    high = layers.Multiply(name="aspp_mul")([high, gate])
    high = layers.UpSampling2D(2, interpolation="bilinear", name="up8")(high)

    mid = _conv_bn_relu(taps[8], decoder_filters, "skip8")
    x = layers.Add(name="fuse8")([high, mid])

    if output_stride == 4:
        x = layers.UpSampling2D(2, interpolation="bilinear", name="up4")(x)
        x = layers.Add(name="fuse4")([x, _conv_bn_relu(taps[4], decoder_filters, "skip4")])

    x = layers.SeparableConv2D(decoder_filters, 3, padding="same", use_bias=False, name="head_conv")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.ReLU(6.0, name="head_relu")(x)
    x = layers.Conv2D(num_outputs, 1, name="logits")(x)
    outputs = layers.Softmax(name="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=f"segnet_a{alpha}_s{output_stride}")
