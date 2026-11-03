"""Checked against published worked examples, so these tests can't simply mirror the implementation."""
import pytest

from annotation_quality import cohen_kappa, fleiss_kappa, krippendorff_alpha, percent_agreement

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 observers, 12 units, gaps.
K2011 = [
    [1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3],
    [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 2, 1],
    [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1], [None, 3, None, None],
]


def test_cohen_kappa_textbook_table():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no gives po=0.70, pe=0.50, kappa=0.40.
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert percent_agreement(a, b) == pytest.approx(0.70)
    assert cohen_kappa(a, b) == pytest.approx(0.40)


def test_kappa_is_zero_when_agreement_is_only_chance():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_kappa_penalises_high_raw_agreement_on_a_skewed_label():
    a = ["ok"] * 18 + ["bad", "ok"]
    b = ["ok"] * 18 + ["ok", "bad"]
    assert percent_agreement(a, b) == pytest.approx(0.90)
    assert cohen_kappa(a, b) < 0


def test_fleiss_kappa_published_example():
    # Standard worked example (Wikipedia, "Fleiss' kappa"): 14 raters, 10 subjects, 5 categories, kappa = 0.210.
    table = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=0.001)


@pytest.mark.parametrize("level,expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)])
def test_krippendorff_published_values(level, expected):
    assert krippendorff_alpha(K2011, level) == pytest.approx(expected, abs=0.001)


def test_krippendorff_ignores_ratings_that_cannot_be_paired():
    without_lone_rating = [unit for unit in K2011 if sum(v is not None for v in unit) >= 2]
    assert krippendorff_alpha(K2011) == krippendorff_alpha(without_lone_rating)


def test_krippendorff_binary_published_value():
    # Krippendorff (2011) binary example: two observers, ten units, alpha = 0.095.
    a = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    assert krippendorff_alpha(list(map(list, zip(a, b)))) == pytest.approx(0.095, abs=0.001)


def test_perfect_agreement_and_constant_labels():
    assert krippendorff_alpha([["a", "a"], ["b", "b"]]) == pytest.approx(1.0)
    assert krippendorff_alpha([["a", "a"], ["a", "a"]]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


@pytest.mark.parametrize("call", [
    lambda: cohen_kappa(["a"], ["a", "b"]),
    lambda: cohen_kappa([], []),
    lambda: fleiss_kappa([]),
    lambda: fleiss_kappa([[1, 1], [2, 1]]),
    lambda: fleiss_kappa([[1, 0]]),
    lambda: krippendorff_alpha([["a", None]]),
    lambda: krippendorff_alpha([["a", "b"]], "ratio-ish"),
])
def test_invalid_input_is_rejected(call):
    with pytest.raises(ValueError):
        call()
