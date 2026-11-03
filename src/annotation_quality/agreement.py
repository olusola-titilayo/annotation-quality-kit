"""Agreement statistics. Each one corrects raw agreement for what chance alone would produce."""
from collections import Counter
from itertools import permutations

_LEVELS = ("nominal", "ordinal", "interval")


def percent_agreement(a, b):
    """Share of items two annotators labelled the same. Not corrected for chance."""
    _check_pair(a, b)
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a, b):
    """Cohen's kappa for two annotators who labelled the same items."""
    _check_pair(a, b)
    n = len(a)
    observed = percent_agreement(a, b)
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[c] * count_b[c] for c in count_a) / (n * n)
    if expected == 1:
        return 1.0  # both annotators used a single, identical label: kappa is otherwise 0/0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(table):
    """Fleiss' kappa. ``table[i][j]`` = raters who put item i in category j; same rater count per item."""
    if not table or not table[0]:
        raise ValueError("table must not be empty")
    raters = sum(table[0])
    if raters < 2:
        raise ValueError("Fleiss' kappa needs at least two ratings per item")
    if any(len(row) != len(table[0]) for row in table):
        raise ValueError("every row needs the same number of categories")
    if any(sum(row) != raters for row in table):
        raise ValueError("every item needs the same number of ratings")

    items = len(table)
    per_item = [(sum(c * c for c in row) - raters) / (raters * (raters - 1)) for row in table]
    observed = sum(per_item) / items
    totals = [sum(row[j] for row in table) for j in range(len(table[0]))]
    expected = sum((t / (items * raters)) ** 2 for t in totals)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def krippendorff_alpha(units, level="nominal"):
    """Krippendorff's alpha. ``units`` holds one list of ratings per item; ``None`` marks a missing rating.

    Handles any number of annotators and missing ratings. Items with fewer than two ratings cannot be
    compared and are dropped, as in Krippendorff (2011).
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}")
    pairable = [[v for v in unit if v is not None] for unit in units]
    pairable = [unit for unit in pairable if len(unit) >= 2]
    if not pairable:
        raise ValueError("no item has two or more ratings")

    coincidences = Counter()
    for unit in pairable:
        for c, k in permutations(unit, 2):
            coincidences[(c, k)] += 1 / (len(unit) - 1)

    marginals = Counter()
    for (c, _), weight in coincidences.items():
        marginals[c] += weight
    total = sum(marginals.values())
    delta = _distance(level, marginals)

    observed = sum(weight * delta(c, k) for (c, k), weight in coincidences.items())
    expected = sum(marginals[c] * marginals[k] * delta(c, k) for c in marginals for k in marginals) / (total - 1)
    if expected == 0:
        return 1.0  # every rating is identical, so there is no variation to explain
    return 1 - observed / expected


def _distance(level, marginals):
    if level == "nominal":
        return lambda c, k: 0.0 if c == k else 1.0
    if level == "interval":
        return lambda c, k: float(c - k) ** 2

    ranks = sorted(marginals)
    position = {value: i for i, value in enumerate(ranks)}

    def ordinal(c, k):
        low, high = sorted((position[c], position[k]))
        between = sum(marginals[ranks[i]] for i in range(low, high + 1))
        return (between - (marginals[ranks[low]] + marginals[ranks[high]]) / 2) ** 2

    return ordinal


def _check_pair(a, b):
    if len(a) != len(b):
        raise ValueError("both annotators must label the same items")
    if not a:
        raise ValueError("no items to compare")
