"""Docker build step: download the MobileNetV2 ImageNet weights into a folder.

Studio runs learning blocks without network access, so every backbone width
the block offers is baked into the image. Usage: fetch_weights.py <dir>
"""

import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seg.model import ALPHAS, WEIGHTS_URL, weights_name  # noqa: E402

out = Path(sys.argv[1]).resolve()
for alpha in ALPHAS:
    name = weights_name(alpha)
    tf.keras.utils.get_file(name, WEIGHTS_URL + name, cache_dir=str(out.parent), cache_subdir=out.name)
    print("fetched", out / name)
