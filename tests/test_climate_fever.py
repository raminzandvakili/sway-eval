import pytest

from sway_eval.loaders.climate_fever import item_from_row
from sway_eval.units import render

ROW = {
    "claim_id": "7", "claim": "A claim.", "claim_label": 3,
    "evidences": [
        {"evidence_id": f"Art{i}:{i}", "evidence_label": lab, "article": f"Art{i}",
         "evidence": f"Sentence {i}.", "entropy": ent, "votes": []}
        for i, (lab, ent) in enumerate([(0, 0.0), (1, 0.0), (2, 0.69), (0, 0.5), (1, 1.0)])
    ],
}


def test_row_to_item():
    it = item_from_row(ROW)
    assert it.id == "cf-7" and len(it.units) == 5
    assert it.meta["claim_label"] == "DISPUTED"
    assert it.meta["net_evidence"] == 0
    assert it.meta["net_evidence_weighted"] == pytest.approx(1.0 - 1.0 + 0.5 - 0.0)
    text = render(it, it.convention_order)
    assert "Art0:" in text and "Sentence 4." in text


def test_parquet_round_trip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from sway_eval.loaders.climate_fever import load
    pq.write_table(pa.Table.from_pylist([ROW]), tmp_path / "t.parquet")
    assert load(tmp_path / "t.parquet")[0].meta["evidence_labels"][:2] == ["SUPPORTS", "REFUTES"]
