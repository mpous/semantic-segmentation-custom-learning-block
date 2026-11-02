"""Classes: how labeler classes map to model output channels.

The labeler has three classes by default: ``floor`` (the default class, never
written as a box), ``obstacle`` and ``ignore``. Studio only knows the labels
that appear on boxes, so its label list is e.g. ``["ignore", "obstacle"]``.

The model output follows the FOMO convention so Edge Impulse can deploy it:
channel 0 is the background (our default class, ``floor``) and channel
``i + 1`` is Studio label ``i``. ``ignore`` keeps its channel when Studio has
it (the deployment expects one per label) but is never a training target:

* ``exclude``: cells that are mostly ``ignore`` get loss weight 0;
* ``obstacle``: ``ignore`` counts as ``obstacle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

IGNORE_MODES = ("exclude", "obstacle")


@dataclass(frozen=True)
class ClassSpec:
    #: Labeler classes; index = class id in the decoded grid.
    names: tuple[str, ...]
    default_class: int
    #: Studio's label list, in Studio's order (output channel = index + 1).
    studio_labels: tuple[str, ...]
    ignore_class: int | None
    #: Class that ``ignore`` is merged into in ``obstacle`` mode.
    obstacle_class: int | None
    ignore_mode: str

    @property
    def num_channels(self) -> int:
        return len(self.studio_labels) + 1

    @property
    def channel_names(self) -> list[str]:
        return [self.names[self.default_class], *self.studio_labels]

    def channel_of(self, k: int) -> int:
        """Output channel for labeler class ``k``."""
        if k == self.default_class:
            return 0
        return self.studio_labels.index(self.names[k]) + 1

    @property
    def trained_channels(self) -> list[int]:
        """Channels that are ever a target: all but ``ignore``'s, if it has one."""
        ignore = None
        if self.ignore_class is not None and self.names[self.ignore_class] in self.studio_labels:
            ignore = self.channel_of(self.ignore_class)
        return [c for c in range(self.num_channels) if c != ignore]


def make_spec(
    names: Sequence[str],
    default_class: str,
    studio_labels: Sequence[str],
    ignore_class: str | None = "ignore",
    obstacle_class: str | None = "obstacle",
    ignore_mode: str = "exclude",
) -> ClassSpec:
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError(f"Class names must be unique: {names}")
    if default_class not in names:
        raise ValueError(f'Default class "{default_class}" is not in the class list {list(names)}')
    if ignore_mode not in IGNORE_MODES:
        raise ValueError(f"ignore mode must be one of {IGNORE_MODES}, not {ignore_mode!r}")

    studio_labels = tuple(studio_labels)
    if default_class in studio_labels:
        raise ValueError(
            f'Studio has boxes labeled "{default_class}", the default class. The labeler never '
            "writes those; remove them or change the default class.")
    unknown = [s for s in studio_labels if s not in names]
    if unknown:
        raise ValueError(
            f"Studio labels {unknown} are not labeler classes {list(names)}. "
            "Set the class list in the block parameters to match the extension's settings.")

    ig = names.index(ignore_class) if ignore_class and ignore_class in names else None
    ob = names.index(obstacle_class) if obstacle_class and obstacle_class in names else None
    if ig is not None and ignore_mode == "obstacle" and ob is None:
        raise ValueError(f'ignore mode "obstacle" needs a class named "{obstacle_class}"')
    for k, n in enumerate(names):
        if k != names.index(default_class) and n not in studio_labels and k != ig:
            # A class nobody has used yet: it can't be deployed without a channel.
            raise ValueError(
                f'Class "{n}" has no boxes in this project yet, so the model has no output for it. '
                "Label at least one sample with it, or remove it from the class list.")
    return ClassSpec(names, names.index(default_class), studio_labels, ig, ob, ignore_mode)
