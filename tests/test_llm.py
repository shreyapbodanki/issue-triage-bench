import json

import pytest

from bench.data import LABELS, stratified_split
from bench.methods.llm import ClaudeClassifier, FewShotSelector, parse_label
from fixtures import FakeClient, make_issues

PRICES = {"fake-model": {"input": 1.0, "output": 5.0}}


@pytest.mark.parametrize("text,label", [
    ("bug", "bug"), ("Bug", "bug"), (" feature.", "feature"), ("Question", "question"),
    ("It is a bug", "bug"), ("debugging", "invalid"), ("", "invalid"), (None, "invalid"),
])
def test_parse_label(text, label):
    assert parse_label(text) == label


def _split():
    return stratified_split(make_issues(30))


def test_random_selector_round_robin():
    train, _, test = _split()
    sel = FewShotSelector(train, 2, "random")
    ex = sel.pick(test[0])
    assert [e["label"] for e in ex] == LABELS * 2
    # same examples for every issue so results are comparable
    assert [e["id"] for e in ex] == [e["id"] for e in sel.pick(test[1])]


def test_retrieval_selector_picks_similar_and_never_itself():
    train, _, _ = _split()
    sel = FewShotSelector(train, 1, "retrieval")
    rec = train[0]
    ex = sel.pick(rec)
    assert len(ex) == 3 and rec["id"] not in {e["id"] for e in ex}
    same = [e for e in ex if e["label"] == rec["label"]][0]
    assert set(same["title"].lower().split()) & set(rec["title"].lower().split())


def test_messages_alternate_and_end_with_target(tmp_path):
    train, _, test = _split()
    clf = ClaudeClassifier("fake-model", PRICES, shots=1, train=train,
                           client=FakeClient(), cache_dir=tmp_path)
    msgs = clf.build_messages(test[0])
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3 + ["user"]
    assert test[0]["title"] in msgs[-1]["content"]
    assert all(m["content"] in LABELS for m in msgs if m["role"] == "assistant")


def test_cache_means_second_run_is_free(tmp_path):
    train, _, test = _split()
    client = FakeClient()
    clf = ClaudeClassifier("fake-model", PRICES, client=client, cache_dir=tmp_path, workers=2)
    first = clf.run(test[:10], progress=False)
    assert client.calls == 10 and not any(r["cached"] for r in first)
    clf2 = ClaudeClassifier("fake-model", PRICES, client=client, cache_dir=tmp_path)
    second = clf2.run(test[:10], progress=False)
    assert client.calls == 10 and all(r["cached"] for r in second)
    assert [r["pred"] for r in first] == [r["pred"] for r in second]


def test_retries_on_overloaded(tmp_path, monkeypatch):
    monkeypatch.setattr("bench.methods.llm.time.sleep", lambda s: None)
    _, _, test = _split()
    client = FakeClient(fail_first=2)
    clf = ClaudeClassifier("fake-model", PRICES, client=client, cache_dir=tmp_path, workers=1)
    out = clf.run(test[:1], progress=False)
    assert client.calls == 3 and out[0]["pred"] in LABELS


def test_turns_thinking_off_and_sends_no_temperature(tmp_path):
    _, _, test = _split()
    client = FakeClient()
    clf = ClaudeClassifier("fake-model", PRICES, client=client, cache_dir=tmp_path, workers=1)
    clf.run(test[:1], progress=False)
    assert client.last_kwargs == {"thinking": {"type": "disabled"}}


def test_falls_back_when_model_rejects_thinking_setting(tmp_path):
    _, _, test = _split()
    client = FakeClient(reject_thinking=True)
    clf = ClaudeClassifier("fake-model", PRICES, client=client, cache_dir=tmp_path, workers=1)
    out = clf.run(test[:2], progress=False)
    assert all(r["pred"] in LABELS for r in out)
    assert client.last_kwargs == {}


def test_cost_uses_prices(tmp_path):
    clf = ClaudeClassifier("fake-model", PRICES, client=FakeClient(), cache_dir=tmp_path)
    assert clf.cost(1_000_000, 1_000_000) == pytest.approx(6.0)


def test_unknown_model_price_fails(tmp_path):
    with pytest.raises(ValueError):
        ClaudeClassifier("nope", PRICES, client=FakeClient(), cache_dir=tmp_path)


def test_real_prices_file_has_defaults():
    prices = json.load(open("config/prices.json"))
    assert "claude-haiku-5-5" in prices and "claude-sonnet-5-5" in prices
