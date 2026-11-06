"""Climate-FEVER (Diggelmann et al., 2020) as records of five evidence units.

Each claim comes with five evidence sentences from Wikipedia. Each sentence
becomes one unit, headed by its article title so it reads on its own. The
file's order is kept as the convention order; whether it is the retrieval
rank order is not documented, so check before treating it as such.

Label names follow the dataset card: claims are SUPPORTS, REFUTES,
NOT_ENOUGH_INFO or DISPUTED; evidence sentences are SUPPORTS, REFUTES or
NOT_ENOUGH_INFO.
"""

from __future__ import annotations

from pathlib import Path

from ..units import Item, Unit

CLAIM_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")
EVIDENCE_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")


def item_from_row(row: dict) -> Item:
    """One claim as an Item. ``meta['net_evidence']`` is #SUPPORTS - #REFUTES;
    ``meta['net_evidence_weighted']`` weights each by (1 - entropy);
    ``meta['stance_count']`` is #SUPPORTS + #REFUTES."""
    units, labels, net, net_w = [], [], 0, 0.0
    for ev in row["evidences"]:
        label = EVIDENCE_LABELS[ev["evidence_label"]]
        units.append(Unit(id=ev["evidence_id"], heading=ev["article"], lines=(ev["evidence"],)))
        labels.append(label)
        sign = 1 if label == "SUPPORTS" else -1 if label == "REFUTES" else 0
        net += sign
        net_w += sign * (1.0 - float(ev["entropy"]))
    return Item(
        id=f"cf-{row['claim_id']}",
        units=tuple(units),
        convention_order=tuple(u.id for u in units),
        meta={
            "claim": row["claim"],
            "claim_label": CLAIM_LABELS[row["claim_label"]],
            "gold": CLAIM_LABELS[row["claim_label"]],
            "stance_count": sum(1 for lab in labels if lab != "NOT_ENOUGH_INFO"),
            "evidence_labels": labels,
            "net_evidence": net,
            "net_evidence_weighted": net_w,
            "prompt_fields": {"claim": row["claim"]},
        },
    )


def load(parquet_path: str | Path) -> list[Item]:
    """Read the dataset's parquet file (needs the ``data`` extra)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError("reading parquet needs the data extra: uv sync --extra data") from e
    rows = pq.read_table(str(parquet_path)).to_pylist()
    items = [item_from_row(r) for r in rows]
    if any(len(it.units) != 5 for it in items):
        raise ValueError("expected exactly five evidence sentences per claim")
    if len(set(it.id for it in items)) != len(items):
        raise ValueError("duplicate claim ids")
    return items
