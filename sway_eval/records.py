"""Reading and writing items and call records as JSON lines."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .units import Item, Unit


def item_to_dict(item: Item) -> dict:
    return {
        "id": item.id,
        "units": [{"id": u.id, "heading": u.heading, "lines": list(u.lines)} for u in item.units],
        "convention_order": list(item.convention_order),
        "meta": item.meta,
    }


def item_from_dict(d: dict) -> Item:
    return Item(
        id=d["id"],
        units=tuple(Unit(id=u["id"], heading=u.get("heading"), lines=tuple(u["lines"])) for u in d["units"]),
        convention_order=tuple(d["convention_order"]),
        meta=d.get("meta", {}),
    )


def save_items(items: Iterable[Item], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(item_to_dict(it), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_items(path: str | Path) -> list[Item]:
    return [item_from_dict(d) for d in read_jsonl(path)]


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_items(items: list[Item], n: int, seed: int | str, stratify: str | None = None) -> list[Item]:
    """A seeded subsample of ``n`` items, in their original order. With
    ``stratify``, each value of ``item.meta[stratify]`` gets a share of ``n``
    proportional to its size (largest remainders break ties in value order)."""
    import random

    if not 0 < n <= len(items):
        raise ValueError(f"cannot sample {n} of {len(items)} items")
    if stratify is None:
        keep = set(random.Random(f"{seed}|sample").sample([it.id for it in items], n))
        return [it for it in items if it.id in keep]
    groups: dict[str, list[str]] = {}
    for it in items:
        groups.setdefault(str(it.meta.get(stratify)), []).append(it.id)
    quota = {g: n * len(ids) / len(items) for g, ids in sorted(groups.items())}
    take = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: (-(quota[g] - take[g]), g))[: n - sum(take.values())]:
        take[g] += 1
    keep = set()
    for g, ids in sorted(groups.items()):
        keep |= set(random.Random(f"{seed}|sample|{g}").sample(ids, take[g]))
    return [it for it in items if it.id in keep]
