"""End to end on a tiny synthetic project laid out the way Studio lays it out."""

import json

import numpy as np
import tensorflow as tf

import train
from seg.export import INT8_FILE, run_tflite

W, H, COLS, ROWS = 64, 48, 16, 12


def runboxes(grid, labels):
    """Minimal run-box encoder (rows of equal class -> boxes), like the extension's."""
    cw, ch = W / COLS, H / ROWS
    boxes = []
    for r in range(ROWS):
        c = 0
        while c < COLS:
            v, start = grid[r, c], c
            while c < COLS and grid[r, c] == v:
                c += 1
            if v:
                boxes.append({"label": labels.index({1: "obstacle", 2: "ignore"}[v]) + 1,
                              "x": round(start * cw), "y": round(r * ch),
                              "w": round((c - start) * cw), "h": round(ch)})
    return boxes


def make_project(path, n, seed, labels):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        grid = np.zeros((ROWS, COLS), np.uint8)
        img = np.full((H, W, 3), (0.6, 0.55, 0.5), np.float32)       # floor
        r0, c0 = rng.integers(0, ROWS - 4), rng.integers(0, COLS - 4)
        grid[r0:r0 + 4, c0:c0 + 4] = 1
        img[r0 * 4:(r0 + 4) * 4, c0 * 4:(c0 + 4) * 4] = (0.1, 0.1, 0.8)  # obstacle
        grid[0, :] = 2                                                  # ignore strip
        xs.append(img)
        ys.append({"sampleId": seed * 1000 + i, "boundingBoxes": runboxes(grid, labels)})
    return np.stack(xs), ys


def test_train_end_to_end(tmp_path):
    labels = ["ignore", "obstacle"]
    data = tmp_path / "data"
    data.mkdir()
    for split, n, seed in (("train", 24, 1), ("test", 8, 2)):
        x, y = make_project(data, n, seed, labels)
        np.save(data / f"X_split_{split}.npy", x)
        (data / f"Y_split_{split}.npy").write_text(json.dumps(y))
    (tmp_path / "train_input.json").write_text(json.dumps({"classes": labels, "inputShapeString": f"({H}, {W}, 3)"}))

    out = tmp_path / "out"
    assert train.main([
        "--data-directory", str(data), "--out-directory", str(out),
        "--info-file", str(tmp_path / "train_input.json"), "--custom-data-directory", str(tmp_path / "cd"),
        "--epochs", "3", "--pretrained", "0", "--grid-cols", str(COLS), "--grid-rows", str(ROWS),
        "--batch-size", "8",
    ]) == 0

    for f in ("model.tflite", INT8_FILE, "saved_model.zip", "metrics.json"):
        assert (out / f).is_file(), f
    report = json.loads((tmp_path / "cd" / "metrics.json").read_text())
    assert report["channels"] == ["floor", "ignore", "obstacle"]
    assert report["output"] == {"width": W // 8, "height": H // 8, "stride": 8}
    assert {"float", "int8", "per_sample"} <= report.keys()
    assert len(report["per_sample"]) == 8

    # FOMO layout: (1, H/8, W/8, labels + 1), int8 in and out.
    interp = tf.lite.Interpreter(model_path=str(out / INT8_FILE))
    assert interp.get_input_details()[0]["dtype"] == np.int8
    assert list(interp.get_output_details()[0]["shape"]) == [1, H // 8, W // 8, 3]
    probs = run_tflite(out / INT8_FILE, np.zeros((1, H, W, 3), np.uint8))
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=0.02)
