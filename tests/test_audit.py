import pytest

from annotation_quality import adjudication_queue, annotator_report, load_labels
from annotation_quality.cli import main

LABELS = {
    "q1": {"ann": "good", "ben": "good", "cal": "good"},
    "q2": {"ann": "good", "ben": "good", "cal": "bad"},
    "q3": {"ann": "bad", "ben": "bad", "cal": "good"},
    "q4": {"ann": "good", "ben": "bad"},
}


def by_name(report):
    return {row["annotator"]: row for row in report}


def test_peer_agreement_excludes_the_annotators_own_vote():
    report = by_name(annotator_report(LABELS))
    assert report["cal"]["peer_agreement"] == pytest.approx(1 / 3)
    # ann: q1 agrees with peers, q4 disagrees with the only peer, q2 and q3 are peer ties and not counted.
    assert report["ann"]["peer_agreement"] == pytest.approx(1 / 2)


def test_tied_peers_are_not_counted():
    report = by_name(annotator_report({"q": {"a": "x", "b": "x", "c": "y", "d": "z"}}))
    # c's peers split x,x,z (clear majority x); a's peers split x,y,z (tie, so not counted).
    assert report["c"]["peer_agreement"] == 0
    assert report["a"]["peer_agreement"] is None


def test_gold_accuracy():
    report = by_name(annotator_report(LABELS, {"q1": "good", "q3": "good"}))
    assert report["cal"]["gold_accuracy"] == 1
    assert report["ann"]["gold_accuracy"] == 0.5
    assert report["ann"]["gold_items"] == 2


def test_queue_orders_gold_contradictions_first_then_weakest_share():
    queue = adjudication_queue(LABELS, gold={"q3": "good"})
    assert [q["item"] for q in queue] == ["q3", "q4", "q2"]
    assert queue[0]["contradicts_gold"] is True
    assert queue[1]["majority"] is None  # 1-1 split has no majority


def test_unanimous_wrong_answer_is_still_queued():
    queue = adjudication_queue({"q": {"a": "x", "b": "x"}}, gold={"q": "y"})
    assert queue[0]["contradicts_gold"] is True


def test_load_rejects_conflicting_duplicates(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("item,annotator,label\nq1,ann,good\nq1,ann,bad\n")
    with pytest.raises(ValueError, match="two different labels"):
        load_labels(path)


def test_load_rejects_missing_columns(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("item,label\nq1,good\n")
    with pytest.raises(ValueError, match="annotator"):
        load_labels(path)


def test_cli_quality_gate(tmp_path, capsys):
    path = tmp_path / "labels.csv"
    path.write_text("item,annotator,label\nq1,a,x\nq1,b,y\nq2,a,y\nq2,b,x\n")
    assert main([str(path)]) == 0
    assert main([str(path), "--min-alpha", "0.667"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_reports_bad_input_with_exit_2(tmp_path):
    assert main([str(tmp_path / "missing.csv")]) == 2
