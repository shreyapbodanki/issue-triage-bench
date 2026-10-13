import json

import pytest

from bench import evaluate, run
from fixtures import FakeClient, write_fixture


def test_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "json", json)
    data, res = tmp_path / "data", tmp_path / "results"
    write_fixture(data)

    run.main(["tfidf", "--data", str(data), "--out", str(res)])
    # point the price lookup at a fake model through the real prices file
    client = FakeClient()
    run.main(["claude", "--model", "claude-haiku-5-5", "--data", str(data), "--out", str(res)], client=client)
    run.main(["claude", "--model", "claude-haiku-5-5", "--shots", "2", "--select", "retrieval",
              "--data", str(data), "--out", str(res)], client=client)
    evaluate.main(["--results", str(res), "--data", str(data)])

    summary = json.load(open(res / "summary.json"))
    assert set(summary) == {"tfidf-logreg", "claude-haiku-5-5-zeroshot", "claude-haiku-5-5-2shot-retrieval"}
    for s in summary.values():
        assert 0 <= s["macro_f1"] <= 1
        lo, hi = s["macro_f1_ci"]
        assert lo <= s["macro_f1"] + 1e-9 and s["macro_f1"] <= hi + 1e-9
    assert summary["tfidf-logreg"]["cost_usd"] == 0
    assert summary["claude-haiku-5-5-zeroshot"]["cost_usd"] > 0
    assert "vs_baseline" in summary["claude-haiku-5-5-zeroshot"]
    report = (res / "report.md").read_text()
    assert "| method |" in report and "leakage" in report
    assert (res / "f1_vs_cost.png").exists() and (res / "confusion.png").exists()
    assert (res / "errors" / "tfidf-logreg.md").exists()


def test_cost_cap_stops_expensive_runs(tmp_path):
    data, res = tmp_path / "data", tmp_path / "results"
    write_fixture(data)
    with pytest.raises(SystemExit):
        run.main(["claude", "--model", "claude-opus-5-5", "--shots", "5", "--max-cost", "0.000001",
                  "--data", str(data), "--out", str(res)], client=FakeClient())
