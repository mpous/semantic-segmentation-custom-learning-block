import numpy as np
import pytest
import tensorflow as tf

from seg.classes import make_spec
from seg.targets import build_targets, grid_fractions

NAMES = ["floor", "obstacle", "ignore"]


def cell(*fractions):
    """One output cell with the given floor/obstacle/ignore fractions."""
    return tf.constant([[list(fractions)]], tf.float32)


def run(spec, frac, threshold=0.25):
    target, weight = build_targets(frac, spec, threshold)
    return target.numpy()[0, 0].tolist(), float(weight.numpy()[0, 0])


@pytest.fixture
def exclude():
    return make_spec(NAMES, "floor", ["ignore", "obstacle"], ignore_mode="exclude")


@pytest.fixture
def merge():
    return make_spec(NAMES, "floor", ["ignore", "obstacle"], ignore_mode="obstacle")


def test_channels_follow_studio_order(exclude):
    # Studio sorts labels; channel 0 is the background (floor).
    assert exclude.channel_names == ["floor", "ignore", "obstacle"]
    assert exclude.channel_of(1) == 2


def test_threshold_leans_to_obstacle(exclude):
    assert run(exclude, cell(0.8, 0.2, 0.0)) == ([1, 0, 0], 1.0)
    assert run(exclude, cell(0.75, 0.25, 0.0)) == ([0, 0, 1], 1.0)


def test_exclude_zeroes_mostly_ignored_cells(exclude):
    assert run(exclude, cell(0.4, 0.0, 0.6))[1] == 0.0
    # Threshold is a share of the labeled part: 0.2 of 0.6 labeled = 33%.
    assert run(exclude, cell(0.4, 0.2, 0.4)) == ([0, 0, 1], 1.0)


def test_obstacle_mode_merges_ignore(merge):
    assert run(merge, cell(0.4, 0.0, 0.6)) == ([0, 0, 1], 1.0)
    assert run(merge, cell(0.9, 0.0, 0.1)) == ([1, 0, 0], 1.0)


def test_ignore_channel_is_never_a_target(exclude, merge):
    rng = np.random.default_rng(0)
    frac = rng.dirichlet([1, 1, 1], size=(1, 30, 40)).astype(np.float32)
    for spec in (exclude, merge):
        target, _ = build_targets(tf.constant(frac), spec, 0.25)
        assert target.numpy()[..., 1].sum() == 0


def test_area_fractions():
    grid = np.zeros((1, 8, 8), np.int32)
    grid[0, :4, :2] = 1          # top-left output cell: half obstacle
    frac = grid_fractions(tf.one_hot(grid, 3), 2, 2).numpy()
    np.testing.assert_allclose(frac[0, 0, 0], [0.5, 0.5, 0.0])


def test_spec_errors():
    with pytest.raises(ValueError, match="not labeler classes"):
        make_spec(NAMES, "floor", ["chair"])
    with pytest.raises(ValueError, match="default class"):
        make_spec(NAMES, "floor", ["floor", "obstacle"])
    with pytest.raises(ValueError, match="no boxes"):
        make_spec(["floor", "obstacle", "person"], "floor", ["obstacle"])
    # ignore unused yet: fine, the model just has no ignore channel.
    assert make_spec(NAMES, "floor", ["obstacle"]).channel_names == ["floor", "obstacle"]
