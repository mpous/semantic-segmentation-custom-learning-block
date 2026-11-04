"""Run a trained model on images and save the predicted mask over each one.

Studio's live classification and model testing treat the output as FOMO and
show one merged box per class, so this is the way to see the actual mask.
Download ``model_quantized_int8_io.tflite`` (or ``model.tflite``) from the
project dashboard, then:

    .venv/bin/python scripts/visualize.py model_quantized_int8_io.tflite photos/ --out out/vis

Images are squashed to the model's input size like the impulse does. Each
output PNG shows the input on the left and the mask on the right: red =
obstacle, yellow = ignore (only when the model has that channel), grey grid
= output cells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seg.export import run_tflite  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COLOURS = {1: (230, 40, 60), 2: (250, 200, 30)}  # channel -> RGB; channel 0 (floor) is left clear
ALPHA = 0.5
SCALE = 2  # draw at twice the model input, so 40x40 cells are easy to see


def image_files(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONS)
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"{p} not found")
    if not files:
        raise SystemExit("No images found")
    return files


def load(path: Path, h: int, w: int, c: int) -> np.ndarray:
    """uint8 (h, w, c), squashed to the model input like the impulse's Squash resize."""
    img = tf.io.decode_image(path.read_bytes(), channels=3, expand_animations=False)
    img = tf.image.resize(img, (h, w), method="bilinear", antialias=True)
    if c == 1:
        img = tf.image.rgb_to_grayscale(img)
    return np.clip(np.round(img.numpy()), 0, 255).astype(np.uint8)


def classes(probs: np.ndarray, threshold: float | None) -> np.ndarray:
    """(h, w) channel per cell: argmax, or obstacle wherever its probability >= threshold."""
    cls = probs.argmax(-1)
    if threshold is not None:
        cls = np.where(probs[..., 1] >= threshold, 1, np.where(cls == 1, 0, cls))
    return cls


def render(img: np.ndarray, cls: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    rgb = np.repeat(img, 3, -1) if img.shape[-1] == 1 else img
    rgb = np.kron(rgb, np.ones((SCALE, SCALE, 1), np.uint8)).astype(np.float32)
    cell_h, cell_w = h * SCALE // cls.shape[0], w * SCALE // cls.shape[1]
    up = np.kron(cls, np.ones((cell_h, cell_w), cls.dtype))

    overlay = rgb.copy()
    for ch, colour in COLOURS.items():
        m = up == ch
        overlay[m] = (1 - ALPHA) * overlay[m] + ALPHA * np.array(colour, np.float32)
    overlay[::cell_h, :] = overlay[::cell_h, :] * 0.7 + 255 * 0.3
    overlay[:, ::cell_w] = overlay[:, ::cell_w] * 0.7 + 255 * 0.3

    gap = np.full((rgb.shape[0], 8, 3), 255, np.float32)
    return np.concatenate([rgb, gap, overlay], 1).astype(np.uint8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path, help=".tflite file from the block (int8 or float)")
    p.add_argument("images", nargs="+", help="image files or folders")
    p.add_argument("--out", type=Path, default=Path("out/vis"))
    p.add_argument("--threshold", type=float, default=None,
                   help="mark a cell obstacle at this probability instead of the most likely class "
                        "(lower = more cautious)")
    args = p.parse_args(argv)

    interp = tf.lite.Interpreter(model_path=str(args.model))
    _, h, w, c = interp.get_input_details()[0]["shape"]
    files = image_files(args.images)
    images = np.stack([load(f, h, w, c) for f in files])
    probs = run_tflite(args.model, images)
    print(f"Input {w}x{h}x{c}, output {probs.shape[2]}x{probs.shape[1]}x{probs.shape[3]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for f, img, pr in zip(files, images, probs):
        cls = classes(pr, args.threshold)
        dest = args.out / f"{f.stem}.png"
        dest.write_bytes(tf.io.encode_png(render(img, cls)).numpy())
        print(f"{dest}: obstacle {np.mean(cls == 1):.0%} of cells", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
