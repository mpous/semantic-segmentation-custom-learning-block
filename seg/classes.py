"""Classes: Studio's label list -> model output channels.

The classes are whatever the project is labeled with. Studio passes its label
list (sorted, only labels that appear on boxes) in ``train_input.json``; the
labeler's default class never appears there, because the extension doesn't
write it as boxes: it is every pixel no box covers.

The output follows the FOMO convention so Edge Impulse can deploy it:
channel 0 is the background (the default class) and channel ``i + 1`` is
Studio label ``i``. Decoded label grids use the same ids, so a grid value is
the channel it trains.

An optional *ignore* label (the extension's ignore class) keeps its channel,
since the deployment expects one per label, but is never a training target:
cells that are mostly ignore get loss weight 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClassSpec:
    #: Channel names: background first, then Studio's labels in Studio's order.
    names: tuple[str, ...]
    #: Channel of the ignore label, if Studio has it.
    ignore_class: int | None

    @property
    def studio_labels(self) -> tuple[str, ...]:
        return self.names[1:]

    @property
    def num_channels(self) -> int:
        return len(self.names)

    @property
    def channel_names(self) -> list[str]:
        return list(self.names)

    @property
    def foreground(self) -> list[int]:
        """Channels a cell can be trained as besides the background."""
        return [c for c in range(1, self.num_channels) if c != self.ignore_class]

    @property
    def trained_channels(self) -> list[int]:
        """Channels that are ever a target: all but the ignore label's."""
        return [c for c in range(self.num_channels) if c != self.ignore_class]


def make_spec(studio_labels: Sequence[str], background: str = "background",
              ignore_label: str | None = "ignore") -> ClassSpec:
    labels = tuple(studio_labels)
    if not labels:
        raise ValueError("The project has no labeled boxes. Label some samples with the extension first.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Studio labels must be unique: {list(labels)}")
    background = background.strip() or "background"
    if background in labels:
        raise ValueError(
            f'Studio has boxes labeled "{background}", the name given to the background (default '
            "class). The extension never writes the default class as boxes: set \"Background class "
            "name\" to the extension's default class, or relabel those boxes.")
    ig = labels.index(ignore_label) + 1 if ignore_label and ignore_label in labels else None
    spec = ClassSpec((background, *labels), ig)
    if not spec.foreground:
        raise ValueError(f'Only the ignore label "{ignore_label}" has boxes: there is nothing to learn yet.')
    return spec
