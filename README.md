# Floor / obstacle segmentation — Edge Impulse custom learning block

Trains a small semantic segmentation network (floor vs obstacle) from the
labels drawn with the
[Segmentation Labeler extension](https://github.com/mpous/semantic-segmentation-labeler-ei-extension),
and exports it in the FOMO layout so Edge Impulse can deploy it to the
Arduino UNO Q like any FOMO model.

## How it works

1. **Labels.** The extension stores each mask as one-cell-high bounding boxes
   ("run boxes") on a label grid (default 160×120). `floor` is the default
   class and is never written as boxes. The block decodes the boxes back into
   the grid, using the same cell-centre rule as the extension (checked against
   fixtures the extension generates, `tests/fixtures/runboxes.json`).
2. **Targets.** The grid is area-resized to the model's output (1/8 of the
   input: 40×30 for a 320×240 impulse). A cell is trained as `obstacle` when at
   least *Obstacle threshold* (25%) of its labeled area is obstacle.
   `ignore` either removes the cell from the loss (*exclude*, the default) or
   counts as obstacle (*obstacle*).
3. **Model.** MobileNetV2 (ImageNet weights, width 0.35 by default), cut at
   1/16, with an LR-ASPP decoder back to 1/8. About 135k parameters. Loss: weighted
   cross-entropy (obstacle ×2) + 0.5 × Dice.
4. **Output.** `1 × H/8 × W/8 × (1 + labels)` softmax. Channel 0 is floor,
   then the Studio labels in Studio's order — exactly what FOMO produces, so
   the Edge Impulse SDK, `.eim` files and the Linux runner accept it. The
   `ignore` channel exists, but is never trained.

Exports: `model.tflite` (float), `model_quantized_int8_io.tflite` (int8 in and
out) and `saved_model.zip`.

## Parameters

| Studio name | Flag | Default | Notes |
| --- | --- | --- | --- |
| Number of training cycles | `--epochs` | 60 | |
| Learning rate | `--learning-rate` | 0.001 | Adam, cosine decay |
| Batch size | `--batch-size` | 16 | |
| Backbone width | `--backbone-width` | 0.35 | 0.35 / 0.5 / 0.75 / 1.0 |
| Ignore class | `--ignore-mode` | exclude | `exclude` or `obstacle` |
| Obstacle threshold per cell | `--obstacle-threshold` | 0.25 | lower = more cautious |
| Obstacle loss weight | `--obstacle-weight` | 2.0 | |
| Data augmentation | `--augmentation` | on | flips, zoom crops, lighting, noise |
| Output resolution *(advanced)* | `--output-stride` | 8 | 4 is finer, not verified for deployment |
| Labeler classes *(advanced)* | `--classes` | floor,obstacle,ignore | the extension's order |
| Default class *(advanced)* | `--default-class` | floor | |
| Label grid columns / rows *(advanced)* | `--grid-cols` / `--grid-rows` | 160 / 120 | must match the extension |

The block can't read the extension's settings. If you change the classes
or the grid in the extension, change them here too.

## Metrics

Studio's *Model testing* page treats the model as FOMO and reports a centroid
F1 score, which doesn't measure segmentation. The real numbers are:

- in the **training log** — float and int8 results on the validation set:
  mean IoU, obstacle recall, *false floor* (obstacle predicted as floor, the
  dangerous error) and floor precision;
- in **`metrics.json`**: the same results plus a confusion matrix and per-sample results, worst first. It's
  written next to the models and to the block's custom data folder, where you can download it from the
  training job.

Live classification and *Model testing* also draw the result as FOMO: adjacent obstacle cells
merge into one box with a dot at its centre, so a correct mask looks like a single wrong detection.
To see the mask itself, download `model_quantized_int8_io.tflite` from the dashboard and run:

```bash
.venv/bin/python scripts/visualize.py model_quantized_int8_io.tflite photos/ --out out/vis
```

Each PNG shows the squashed input next to the mask (red = obstacle), one cell per output value.
`--threshold 0.3` marks a cell obstacle from 30% probability, to preview a more cautious setting.

## Run locally

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q

# with data downloaded from a project (edge-impulse-blocks runner --download-data data/)
.venv/bin/python train.py --data-directory data --out-directory out --epochs 30
```

In the Docker image, the ImageNet weights are downloaded at build time, because Studio runs
blocks without network access. To regenerate the decoder fixtures, run `npm run fixtures >
<this repo>/tests/fixtures/runboxes.json` in the extension repository.

## Push to Studio

```bash
npm install -g edge-impulse-cli
edge-impulse-blocks init      # pick your organization, "Machine learning block"
edge-impulse-blocks push
```

Then, in your project: set the labeling method to *Bounding boxes*, create an impulse (Image data
320×320, resize mode **Squash**, Image block, this learning block) and train. Keep the input
square: Studio's FOMO tools (live classification, model testing) reject other shapes.

## Limitations

- The `.eim` and the Linux runner merge obstacle cells into boxes; they don't return the mask. For
  the full mask, run `model_quantized_int8_io.tflite` with LiteRT and read the output tensor.
- Squash resize only: the labels cover the whole frame.
- Only tested with RGB and grayscale image blocks at sizes that are multiples of 16.
