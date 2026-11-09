import numpy as np
import pytest
import tensorflow as tf

from seg.classes import make_spec
from seg.targets import build_targets, grid_fractions


def cell(*fractions):
    """One output cell with the given per-class fractions (channel order)."""
    return tf.constant([[list(fractions)]], tf.float32)


def run(spec, frac, threshold=0.25):
    target, weight = build_targets(frac, spec, threshold)
    return target.numpy()[0, 0].tolist(), float(weight.numpy()[0, 0])


@pytest.fixture
def robot():
    # Studio sorts labels: ignore, obstacle. Channel 0 is the background.
    return make_spec(["ignore", "obstacle"], "floor")


@pytest.fixture
def football():
    return make_spec(["ball", "referee", "teamA", "teamB"], "pitch")


def test_channels_follow_studio_order(robot, football):
    assert robot.channel_names == ["floor", "ignore", "obstacle"]
    assert robot.ignore_class == 1 and robot.foreground == [2]
    assert football.channel_names == ["pitch", "ball", "referee", "teamA", "teamB"]
    assert football.ignore_class is None and football.foreground == [1, 2, 3, 4]


def test_threshold(robot):
    assert run(robot, cell(0.8, 0.0, 0.2)) == ([1, 0, 0], 1.0)
    assert run(robot, cell(0.75, 0.0, 0.25)) == ([0, 0, 1], 1.0)


def test_largest_foreground_class_wins(football):
    # Background has the most, but ball and teamA pass the threshold: teamA is larger.
    assert run(football, cell(0.4, 0.26, 0.0, 0.34, 0.0))[0] == [0, 0, 0, 1, 0]
    assert run(football, cell(0.8, 0.1, 0.1, 0.0, 0.0))[0] == [1, 0, 0, 0, 0]


def test_mostly_ignored_cells_get_no_weight(robot):
    assert run(robot, cell(0.4, 0.6, 0.0))[1] == 0.0
    # Threshold is a share of the labeled part: 0.2 of 0.6 labeled = 33%.
    assert run(robot, cell(0.4, 0.4, 0.2)) == ([0, 0, 1], 1.0)


def test_ignore_channel_is_never_a_target(robot):
    rng = np.random.default_rng(0)
    frac = rng.dirichlet([1, 1, 1], size=(1, 30, 40)).astype(np.float32)
    target, _ = build_targets(tf.constant(frac), robot, 0.25)
    assert target.numpy()[..., 1].sum() == 0


def test_area_fractions():
    grid = np.zeros((1, 8, 8), np.int32)
    grid[0, :4, :2] = 1          # top-left output cell: half class 1
    frac = grid_fractions(tf.one_hot(grid, 3), 2, 2).numpy()
    np.testing.assert_allclose(frac[0, 0, 0], [0.5, 0.5, 0.0])


def test_spec_errors():
    with pytest.raises(ValueError, match="no labeled boxes"):
        make_spec([])
    with pytest.raises(ValueError, match="Background class name"):
        make_spec(["floor", "obstacle"], "floor")
    with pytest.raises(ValueError, match="nothing to learn"):
        make_spec(["ignore"])
    # No ignore label in the project, or none configured: every label trains.
    assert make_spec(["obstacle"], "floor").foreground == [1]
    assert make_spec(["ignore", "obstacle"], "floor", None).foreground == [1, 2]
    assert make_spec(["a"], "  ").channel_names == ["background", "a"]
