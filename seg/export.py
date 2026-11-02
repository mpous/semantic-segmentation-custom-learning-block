"""Model files Studio picks up from ``--out-directory``.

* ``model.tflite``: float32 in and out.
* ``model_quantized_int8_io.tflite``: int8 in and out, calibrated on training
  images. This is what runs on the UNO Q.
* ``saved_model.zip``: the Keras model, for Studio's own conversions.

The int8 file is made here (not left to Studio) so its accuracy can be
measured in the same run (``run_tflite``).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

FLOAT_FILE = "model.tflite"
INT8_FILE = "model_quantized_int8_io.tflite"
SAVED_MODEL_ZIP = "saved_model.zip"


def export_all(model: tf.keras.Model, calibration: np.ndarray, out_dir: Path) -> dict[str, Path]:
    """``calibration``: uint8 images (N, H, W, C) as stored in Split.images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    with tempfile.TemporaryDirectory() as tmp:
        sm = Path(tmp) / "saved_model"
        model.save(sm, include_optimizer=False)
        archive = shutil.make_archive(str(Path(tmp) / "saved_model"), "zip", root_dir=tmp, base_dir="saved_model")
        files["saved_model"] = Path(shutil.move(archive, out_dir / SAVED_MODEL_ZIP))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    files["float"] = _write(out_dir / FLOAT_FILE, converter.convert())

    def representative():
        for img in calibration:
            yield [img[None].astype(np.float32) / 255.0]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    files["int8"] = _write(out_dir / INT8_FILE, converter.convert())
    return files


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def run_tflite(path: Path, images: np.ndarray) -> np.ndarray:
    """Run a .tflite on uint8-stored 0..1 images; returns float probabilities (N, h, w, C)."""
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    results = []
    for img in images:
        x = img[None].astype(np.float32) / 255.0
        if inp["dtype"] == np.int8:
            x = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            y = (y.astype(np.float32) - out_zero) * out_scale
        results.append(y)
    return np.stack(results) if results else np.zeros((0,), np.float32)
