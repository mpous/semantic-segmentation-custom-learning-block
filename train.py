"""Floor / obstacle segmentation learning block: entry point.

Studio runs this with ``--data-directory``, ``--out-directory``,
``--info-file`` and the parameters in parameters.json. See README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from seg import data as D
from seg.classes import IGNORE_MODES, make_spec
from seg.export import export_all, run_tflite
from seg.losses import make_loss, unpack
from seg.metrics import confusion, format_summary, summarize
from seg.model import ALPHAS, build_model


def csv(s: str) -> list[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-directory", required=True)
    p.add_argument("--out-directory", required=True)
    p.add_argument("--info-file", default=None)
    p.add_argument("--custom-data-directory", default="/home/custom_data",
                   help="metrics.json is also written here, for the results viewer")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--backbone-width", type=float, default=0.35, choices=ALPHAS)
    p.add_argument("--output-stride", type=int, default=8, choices=(4, 8))
    p.add_argument("--augmentation", type=int, default=1, choices=(0, 1))
    p.add_argument("--pretrained", type=int, default=1, choices=(0, 1))

    p.add_argument("--classes", type=csv, default=csv("floor,obstacle,ignore"),
                   help="labeler classes in the extension's order (class id = position)")
    p.add_argument("--default-class", default="floor")
    p.add_argument("--ignore-class", default="ignore")
    p.add_argument("--obstacle-class", default="obstacle")
    p.add_argument("--ignore-mode", default="exclude", choices=IGNORE_MODES)
    p.add_argument("--grid-cols", type=int, default=160)
    p.add_argument("--grid-rows", type=int, default=120)
    p.add_argument("--obstacle-threshold", type=float, default=0.25)
    p.add_argument("--obstacle-weight", type=float, default=2.0)
    p.add_argument("--studio-labels", type=csv, default=[],
                   help="local runs only: Studio's label list (Studio passes it in train_input.json)")
    p.add_argument("--seed", type=int, default=1)
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}", flush=True)
    return args


class KeepBest(tf.keras.callbacks.Callback):
    """Restore the weights from the epoch with the lowest validation loss."""

    def __init__(self):
        super().__init__()
        self.best, self.best_epoch, self.weights = math.inf, -1, None

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_loss")
        if v is not None and v < self.best:
            self.best, self.best_epoch, self.weights = v, epoch, self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            print(f"Restoring weights from epoch {self.best_epoch + 1} (val_loss {self.best:.4f})", flush=True)
            self.model.set_weights(self.weights)


def collect_targets(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    ts, ws = [], []
    for _, y in ds:
        t, w = unpack(y)
        ts.append(t.numpy())
        ws.append(w.numpy())
    return np.concatenate(ts), np.concatenate(ws)


def per_sample(sample_ids, target, weight, probs, names, trained):
    rows = []
    for i, sid in enumerate(sample_ids):
        s = summarize(confusion(target[i:i + 1], weight[i:i + 1], probs[i:i + 1]), names, trained)
        rows.append({"sampleId": sid, "mean_iou": s["mean_iou"], "false_floor_rate": s["false_floor_rate"],
                     "accuracy": s["accuracy"]})
    # Worst first: missed obstacles, then low IoU.
    return sorted(rows, key=lambda r: (-(r["false_floor_rate"] or 0), r["mean_iou"] if r["mean_iou"] is not None else 1))


def main(argv=None) -> int:
    args = parse_args(argv)
    tf.keras.utils.set_random_seed(args.seed)
    data_dir, out_dir = Path(args.data_directory), Path(args.out_directory)
    info = D.read_info(args.info_file)
    shape = D.input_shape_from_info(info)

    y_train = D.read_boxes(data_dir / "Y_split_train.npy")
    y_val = D.read_boxes(data_dir / "Y_split_test.npy")
    labels = D.resolve_labels(info, args.studio_labels, args.classes, args.default_class,
                              D.max_label(y_train, y_val))
    spec = make_spec(args.classes, args.default_class, labels, args.ignore_class,
                     args.obstacle_class, args.ignore_mode)

    x_train = D.load_images(data_dir / "X_split_train.npy", shape)
    x_val = D.load_images(data_dir / "X_split_test.npy", shape)
    train = D.decode_split(x_train, y_train, spec, args.grid_cols, args.grid_rows)
    val = D.decode_split(x_val, y_val, spec, args.grid_cols, args.grid_rows) if len(x_val) else None

    n, h, w, c = train.images.shape
    if n == 0:
        raise SystemExit("No training samples")
    out_hw = (h // args.output_stride, w // args.output_stride)
    print(f"Input {w}x{h}x{c}, output {out_hw[1]}x{out_hw[0]} (1/{args.output_stride}), "
          f"channels {spec.channel_names}, ignore mode {spec.ignore_mode}", flush=True)
    print(f"{n} training samples, {len(val.images) if val else 0} validation samples", flush=True)

    model = build_model(h, w, c, spec.num_channels, args.backbone_width, args.output_stride,
                        pretrained=bool(args.pretrained))
    model.summary(print_fn=lambda s: print(s, flush=True), line_length=110)

    batch = min(args.batch_size, n)
    train_ds = D.make_dataset(train, spec, out_hw, args.obstacle_threshold, batch, bool(args.augmentation))
    val_ds = D.make_dataset(val, spec, out_hw, args.obstacle_threshold, batch, False) if val else None

    steps = max(1, math.ceil(n / batch)) * max(1, args.epochs)
    lr = tf.keras.optimizers.schedules.CosineDecay(args.learning_rate, steps, alpha=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_loss(args.obstacle_weight))

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
              callbacks=[KeepBest()] if val_ds else [])
    print(f"Training took {time.time() - t0:.0f} s", flush=True)

    files = export_all(model, train.images[:200], out_dir)
    for k, f in files.items():
        print(f"Wrote {f} ({f.stat().st_size / 1024:.0f} kB)", flush=True)

    report = {
        "v": 1,
        "block": "semantic-segmentation-custom-learning-block",
        "channels": spec.channel_names,
        "ignore_mode": spec.ignore_mode,
        "input": {"width": w, "height": h, "channels": c, "scaling": "0..1"},
        "output": {"width": out_hw[1], "height": out_hw[0], "stride": args.output_stride},
        "label_grid": {"cols": args.grid_cols, "rows": args.grid_rows},
        "params": {k: v for k, v in vars(args).items() if not k.endswith("directory") and k != "info_file"},
    }
    if val:
        trained = spec.trained_channels
        target, weight = collect_targets(val_ds)
        probs = {"float": model.predict(val_ds, verbose=0), "int8": run_tflite(files["int8"], val.images)}
        for kind, p in probs.items():
            report[kind] = summarize(confusion(target, weight, p), spec.channel_names, trained)
            print(format_summary(f"Validation ({kind})", report[kind]), flush=True)
        report["per_sample"] = per_sample(val.sample_ids, target, weight, probs["int8"], spec.channel_names, trained)
    else:
        print("No validation split: skipping evaluation", flush=True)

    text = json.dumps(report, indent=2)
    (out_dir / "metrics.json").write_text(text)
    try:
        cd = Path(args.custom_data_directory)
        cd.mkdir(parents=True, exist_ok=True)
        (cd / "metrics.json").write_text(text)
    except OSError as ex:
        print(f"Could not write custom data ({ex}); metrics.json is in the out directory only", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
