"""Command line report: annotation-quality labels.csv --gold gold.csv"""
import argparse
import csv
import sys

from .agreement import krippendorff_alpha
from .audit import adjudication_queue, annotator_report, load_labels


def _load_gold(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["item"].strip(): row["label"].strip() for row in csv.DictReader(f)}


def _pct(value):
    return "-" if value is None else f"{value:.0%}"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="annotation-quality", description=__doc__)
    parser.add_argument("labels", help="CSV with item,annotator,label columns")
    parser.add_argument("--gold", help="CSV with item,label columns for known-answer items")
    parser.add_argument("--min-alpha", type=float, default=None, help="exit 1 if alpha is below this value")
    parser.add_argument("--threshold", type=float, default=1.0, help="queue items whose top label share is below this")
    args = parser.parse_args(argv)

    try:
        labels = load_labels(args.labels)
        gold = _load_gold(args.gold) if args.gold else {}
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    annotators = sorted({a for votes in labels.values() for a in votes})
    units = [[votes.get(a) for a in annotators] for votes in labels.values()]
    try:
        alpha = krippendorff_alpha(units)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{len(labels)} items, {len(annotators)} annotators")
    print(f"Krippendorff's alpha (nominal): {alpha:.3f}\n")
    print(f"{'annotator':<14}{'items':>7}{'peers':>8}{'gold':>7}")
    for row in annotator_report(labels, gold):
        print(f"{row['annotator']:<14}{row['items']:>7}{_pct(row['peer_agreement']):>8}{_pct(row['gold_accuracy']):>7}")

    queue = adjudication_queue(labels, args.threshold, gold)
    print(f"\nAdjudication queue: {len(queue)} item(s)")
    for q in queue:
        note = f"  CONTRADICTS GOLD ({q['gold']})" if q["contradicts_gold"] else ""
        print(f"  {q['item']:<10} {q['share']:.0%} {q['votes']}{note}")

    if args.min_alpha is not None and alpha < args.min_alpha:
        print(f"\nFAIL: alpha {alpha:.3f} is below the required {args.min_alpha}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
