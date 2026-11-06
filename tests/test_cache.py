from sway_eval.adapters import DecisionRequest, MockAdapter, text_scorer
from sway_eval.cache import CachedAdapter, ResponseCache, cache_key

REQ = DecisionRequest("Decide.", "alpha beta", ("YES", "NO"))


def mock(**kw):
    return MockAdapter(text_scorer({"alpha": 1.0, "beta": -0.5}), [1.0, 1.0], **kw)


def test_hit_after_miss_and_persistence(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    m = mock()
    ca = CachedAdapter(m, cache)
    d1, c1 = ca.decide(REQ)
    d2, c2 = ca.decide(REQ)
    assert (c1, c2) == (False, True) and d1 == d2 and m.calls == 1
    cache.close()
    again = CachedAdapter(mock(), ResponseCache(tmp_path / "c.sqlite"))
    assert again.decide(REQ)[1] is True


def test_repeat_index_is_not_collapsed(tmp_path):
    ca = CachedAdapter(mock(noise=0.5), ResponseCache(tmp_path / "c.sqlite"))
    a, _ = ca.decide(REQ, repeat=1)
    b, _ = ca.decide(REQ, repeat=2)
    assert a.probs != b.probs and ca.misses == 2


def test_key_depends_on_model_and_request():
    assert cache_key("m1", REQ) != cache_key("m2", REQ)
    assert cache_key("m1", REQ) != cache_key("m1", DecisionRequest("Decide.", "beta alpha", ("YES", "NO")))
