# Semantic segmentation — Edge Impulse custom learning block

Trains a small semantic segmentation network, with any classes, from the
labels drawn with the
[Segmentation Labeler extension](https://github.com/mpous/semantic-segmentation-labeler-ei-extension),
and exports it in the FOMO layout so Edge Impulse can deploy it to the
Arduino UNO Q like any FOMO model.

## How it works

1. **Labels.** The extension stores each mask as one-cell-high bounding boxes
   ("run boxes") on a label grid (default 160×120). The extension's default
   class (e.g. `floor`) is never written as boxes: it becomes the *background*.
   **The classes are the project's labels**, read from Studio at training
   time, so nothing has to be listed in the block. The block decodes the boxes back into
   the grid, using the same cell-centre rule as the extension (checked against
   fixtures the extension generates, `tests/fixtures/runboxes.json`).
2. **Targets.** The grid is area-resized to the model's output (1/8 of the
   input: 40×30 for a 320×240 impulse). A cell is trained as a class when at
   least *Class threshold* (25%) of its labeled area has that class (the
   largest such class wins), otherwise as the background. Cells mostly labeled
   with the *Ignore label* (`ignore`, if the project has it) are left out of the loss.
3. **Model.** MobileNetV2 (ImageNet weights, width 0.35 by default), cut at
   1/16, with an LR-ASPP decoder back to 1/8. About 135k parameters. Loss: weighted
   cross-entropy (every class but the background ×2) + 0.5 × Dice, averaged over the classes.
4. **Output.** `1 × H/8 × W/8 × (1 + labels)` softmax. Channel 0 is the background,
   then the Studio labels in Studio's order — exactly what FOMO produces, so
   the Edge Impulse SDK, `.eim` files and the Linux runner accept it. An
   `ignore` channel, if there is one, exists but is never trained.

Exports: `model.tflite` (float), `model_quantized_int8_io.tflite` (int8 in and
out) and `saved_model.zip`.

## Parameters

| Studio name | Flag | Default | Notes |
| --- | --- | --- | --- |
| Number of training cycles | `--epochs` | 60 | |
| Learning rate | `--learning-rate` | 0.001 | Adam, cosine decay |
| Batch size | `--batch-size` | 16 | |
| Backbone width | `--backbone-width` | 0.35 | 0.35 / 0.5 / 0.75 / 1.0 |
| Class threshold per cell | `--class-threshold` | 0.25 | lower keeps small objects |
| Foreground loss weight | `--foreground-weight` | 2.0 | weight of every class but the background |
| Data augmentation | `--augmentation` | on | flips, zoom crops, lighting, noise |
| Output resolution *(advanced)* | `--output-stride` | 8 | 4 is finer, not verified for deployment |
| Background class name *(advanced)* | `--background-name` | background | the extension's default class; only names channel 0 |
| Ignore label *(advanced)* | `--ignore-label` | ignore | blank = every label is trained |
| Label grid columns / rows *(advanced)* | `--grid-cols` / `--grid-rows` | 160 / 120 | must match the extension |

The classes come from Studio, so adding, renaming or removing classes in the
extension needs no change here (retrain after relabeling). The block can't read the
extension's grid: if you change it, change it here too. The training log warns
when the boxes don't look like the configured grid.

## Metrics

Studio's *Model testing* page treats the model as FOMO and reports a centroid
F1 score, which doesn't measure segmentation. The real numbers are:

- in the **training log** — float and int8 results on the validation set:
  IoU per class, mean IoU, *foreground recall* (cells of any class predicted as some
  class), *missed* (objects predicted as background, usually the costly error) and
  background precision;
- in **`metrics.json`**: the same results plus a confusion matrix and per-sample results, worst first. It's
  written next to the models and to the block's custom data folder, where you can download it from the
  training job.

Live classification and *Model testing* also draw the result as FOMO: adjacent cells of a class
merge into one box with a dot at its centre, so a correct mask looks like a single wrong detection.
To see the mask itself, download `model_quantized_int8_io.tflite` from the dashboard and run:

```bash
.venv/bin/python scripts/visualize.py model_quantized_int8_io.tflite photos/ --out out/vis \
    --metrics metrics.json
```

Each PNG shows the squashed input next to the mask (one colour per class, printed at the start),
one cell per output value. `--metrics` (from the training job) or `--labels a,b,c` names the
classes. `--threshold 0.3` marks a cell as an object from 30% probability, to preview a more
cautious setting.

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

- The `.eim` and the Linux runner merge each class's cells into boxes; they don't return the mask. For
  the full mask, run `model_quantized_int8_io.tflite` with LiteRT and read the output tensor.
- Squash resize only: the labels cover the whole frame.
- Only tested with RGB and grayscale image blocks at sizes that are multiples of 16.
