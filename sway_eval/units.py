"""Documents as ordered units, and how units are rendered into a prompt.

A unit is the smallest block of a record that can be moved as a whole and
still reads on its own: a resume section, a dependent claim, an evidence
sentence. The order of units is what the audit varies. Everything else about
how the text looks is set by a ``FormatSpec``, so formatting can be varied
separately as a nuisance baseline.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

_LIST_MARKER = re.compile(r"[-*+\u2022]\s")


class UnitTextWarning(UserWarning):
    """A unit's line breaks the clean-line assumption of ``render_unit``."""


def line_problems(line: str) -> list[str]:
    """Ways a line is not one clean line of text: empty, has a line break,
    has spaces at either end, or starts with its own list marker."""
    problems = []
    if not line.strip():
        problems.append("is empty")
    if "\n" in line or "\r" in line:
        problems.append("has a line break")
    if line != line.strip():
        problems.append("has leading or trailing whitespace")
    if _LIST_MARKER.match(line.lstrip()):
        problems.append("starts with a list marker")
    return problems


@dataclass(frozen=True)
class Unit:
    """One movable block of a record.

    ``heading`` is optional (an evidence sentence may have none, or carry its
    source title). ``lines`` are rendered in the given order; the audit never
    reorders inside a unit.

    Each line should be one clean line of text with no bullet of its own,
    because the format adds bullets (see ``render_unit``). A line that breaks
    this raises a ``UnitTextWarning`` when the unit is built.
    """

    id: str
    lines: tuple[str, ...]
    heading: str | None = None

    def __post_init__(self) -> None:
        for i, line in enumerate(self.lines):
            if not isinstance(line, str):
                continue
            for problem in line_problems(line):
                warnings.warn(f"unit {self.id!r}, line {i} {problem}: {line[:60]!r}", UnitTextWarning, stacklevel=3)


@dataclass(frozen=True)
class Item:
    """A record to decide on: its units, its default order, and metadata.

    ``convention_order`` is the order most real documents of this kind use.
    ``meta`` holds anything the analysis needs, such as a qualification level
    or which unit carries the decisive strength.
    """

    id: str
    units: tuple[Unit, ...]
    convention_order: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def unit(self, unit_id: str) -> Unit:
        for u in self.units:
            if u.id == unit_id:
                return u
        raise KeyError(unit_id)

    def unit_ids(self) -> tuple[str, ...]:
        return tuple(u.id for u in self.units)


@dataclass(frozen=True)
class FormatSpec:
    """How units are turned into text. Changing it never changes content.

    ``heading_style`` is a format string with ``{h}`` for the heading, for
    example ``"{h}:"``, ``"## {h}"`` or ``"{H}"`` (upper case).
    """

    separator: str = "\n\n"
    heading_style: str = "{h}:"
    bullet: str = "- "

    def heading(self, h: str) -> str:
        return self.heading_style.format(h=h, H=h.upper())


DEFAULT_FORMAT = FormatSpec()


def render_unit(unit: Unit, fmt: FormatSpec = DEFAULT_FORMAT) -> str:
    """Text of one unit under a format.

    Assumes each of ``unit.lines`` is one line with no line break and no
    bullet of its own: the bullet is added here, so it stays part of the
    format. A line with a line break would get a bullet on its first part
    only, and one that starts with "- " would render as "- - ...", so a
    format variant would change more than the look. ``Unit`` warns about such
    lines when it is built.
    """
    body = [fmt.bullet + line for line in unit.lines] if fmt.bullet else list(unit.lines)
    if unit.heading:
        return "\n".join([fmt.heading(unit.heading), *body])
    return "\n".join(body)


def render_blank(item: Item, order: Sequence[str], fmt: FormatSpec = DEFAULT_FORMAT) -> str:
    """The record's section headings in ``order`` with no content: the natural
    empty record for subset scoring. Units without a heading render as nothing."""
    if sorted(order) != sorted(item.unit_ids()):
        raise ValueError(f"order {list(order)} is not a permutation of the units of {item.id}")
    return fmt.separator.join(fmt.heading(item.unit(uid).heading) for uid in order if item.unit(uid).heading)


def render(item: Item, order: Sequence[str], fmt: FormatSpec = DEFAULT_FORMAT, subset: bool = False) -> str:
    """Text of a record with its units in ``order``.

    ``order`` must be a permutation of the item's unit ids, so no unit is
    dropped or repeated; with ``subset`` (subset scoring only), any non-empty
    selection of distinct units is allowed.
    """
    if subset:
        if not order or len(set(order)) != len(order) or not set(order) <= set(item.unit_ids()):
            raise ValueError(f"order {list(order)} is not a selection of distinct units of {item.id}")
    elif sorted(order) != sorted(item.unit_ids()):
        raise ValueError(f"order {list(order)} is not a permutation of the units of {item.id}")
    return fmt.separator.join(render_unit(item.unit(uid), fmt) for uid in order)
