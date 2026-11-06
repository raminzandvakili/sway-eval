import pytest

from sway_eval import count_sequences, count_sequences_up_to, exhaustive_calls, ordered_subsets


def test_count_sequences_known_values():
    assert count_sequences(30, 5) == 17_100_720
    assert count_sequences(8, 8) == 40_320
    assert count_sequences(6, 6) == 720
    assert count_sequences(3, 0) == 1
    assert count_sequences(3, 4) == 0


def test_count_matches_enumeration():
    items = list("abcde")
    for k in range(0, 6):
        assert sum(1 for _ in ordered_subsets(items, k)) == count_sequences(5, k)


def test_count_up_to():
    assert count_sequences_up_to(4, 2) == 4 + 12


def test_pilot_budget():
    assert exhaustive_calls(50, 8) == 2_016_000
    assert exhaustive_calls(50, 6) == 36_000


def test_negative_rejected():
    with pytest.raises(ValueError):
        count_sequences(-1, 2)
