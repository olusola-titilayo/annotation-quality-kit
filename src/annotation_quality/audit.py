"""Project-level checks on a labels file: who needs coaching, and which items need a second look."""
import csv
from collections import Counter, defaultdict


def load_labels(path):
    """Read ``item,annotator,label`` rows into ``{item: {annotator: label}}``."""
    labels = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"item", "annotator", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            item, annotator, label = row["item"].strip(), row["annotator"].strip(), row["label"].strip()
            if not (item and annotator and label):
                raise ValueError(f"{path}:{line}: empty item, annotator or label")
            if annotator in labels[item] and labels[item][annotator] != label:
                raise ValueError(f"{path}:{line}: {annotator} gave {item} two different labels")
            labels[item][annotator] = label
    return dict(labels)


def annotator_report(labels, gold=None):
    """Per-annotator counts, agreement with the other annotators' majority, and accuracy on gold items.

    An annotator's own vote is left out of the majority they are compared with, so nobody gets credit
    for agreeing with themselves. Ties among the others are not counted.
    """
    gold = gold or {}
    stats = defaultdict(lambda: {"items": 0, "peer_hits": 0, "peer_total": 0, "gold_hits": 0, "gold_total": 0})
    for item, votes in labels.items():
        for annotator, label in votes.items():
            s = stats[annotator]
            s["items"] += 1
            others = Counter(v for a, v in votes.items() if a != annotator).most_common(2)
            if others and (len(others) == 1 or others[0][1] > others[1][1]):
                s["peer_total"] += 1
                s["peer_hits"] += label == others[0][0]
            if item in gold:
                s["gold_total"] += 1
                s["gold_hits"] += label == gold[item]

    report = []
    for annotator, s in sorted(stats.items()):
        report.append({
            "annotator": annotator,
            "items": s["items"],
            "peer_agreement": s["peer_hits"] / s["peer_total"] if s["peer_total"] else None,
            "gold_accuracy": s["gold_hits"] / s["gold_total"] if s["gold_total"] else None,
            "gold_items": s["gold_total"],
        })
    return report


def adjudication_queue(labels, threshold=1.0, gold=None):
    """Items whose top label share is below ``threshold``, or whose majority contradicts gold. Worst first."""
    gold = gold or {}
    queue = []
    for item, votes in labels.items():
        if len(votes) < 2:
            continue
        counts = Counter(votes.values())
        ranked = counts.most_common()
        share = ranked[0][1] / len(votes)
        tied = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
        majority = None if tied else ranked[0][0]
        contradicts_gold = item in gold and majority != gold[item]
        if share < threshold or contradicts_gold:
            queue.append({
                "item": item,
                "share": share,
                "majority": majority,
                "votes": dict(sorted(counts.items())),
                "gold": gold.get(item),
                "contradicts_gold": contradicts_gold,
            })
    return sorted(queue, key=lambda q: (not q["contradicts_gold"], q["share"], q["item"]))
