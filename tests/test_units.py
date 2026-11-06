import warnings

import pytest

from sway_eval.units import FormatSpec, Item, Unit, UnitTextWarning, line_problems, render

ITEM = Item(
    id="x",
    units=(Unit("a", ("first",), "A"), Unit("b", ("second", "more"), "B"), Unit("c", ("third",))),
    convention_order=("a", "b", "c"),
)


def test_render_orders_units():
    text = render(ITEM, ("c", "a", "b"))
    assert text.index("third") < text.index("first") < text.index("second")


def test_render_rejects_non_permutation():
    with pytest.raises(ValueError):
        render(ITEM, ("a", "a", "b"))


def test_format_changes_presentation_not_content():
    plain = render(ITEM, ITEM.convention_order)
    other = render(ITEM, ITEM.convention_order, FormatSpec(separator="\n---\n", heading_style="## {h}", bullet="* "))
    assert plain != other
    for line in ("first", "second", "more", "third"):
        assert line in plain and line in other
    assert "## A" in other and "A:" in plain


@pytest.mark.parametrize("line, problem", [
    ("", "is empty"),
    ("two\nlines", "has a line break"),
    (" padded", "has leading or trailing whitespace"),
    ("- already bulleted", "starts with a list marker"),
    ("\u2022 already bulleted", "starts with a list marker"),
])
def test_unclean_line_warns(line, problem):
    assert problem in line_problems(line)
    with pytest.warns(UnitTextWarning, match=problem):
        Unit("u", ("fine", line))


def test_clean_lines_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Unit("u", ("5 years of relevant experience.", "-2 degrees is cold.", "A - B"))
