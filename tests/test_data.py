from collections import Counter

import pytest

from bench.data import (balance, clean_body, clean_title, repo_holdout_split,
                        stratified_split, to_text)
from fixtures import make_issues


@pytest.mark.parametrize("raw,clean", [
    ("BUG: groupby drops rows", "groupby drops rows"),
    ("ENH: add axis keyword", "add axis keyword"),
    ("QST: how do I merge", "how do I merge"),
    ("[Bug] crash on start", "crash on start"),
    ("[Feature Request] dark mode", "dark mode"),
    ("[BUG] ENH: double prefix", "double prefix"),
    ("Bugfix needed in parser", "Bugfix needed in parser"),
    ("[DevTools Bug] profiler crashes", "profiler crashes"),
    ("[Compiler Bug]: memo breaks refs", "memo breaks refs"),
    ("[Feature]: allow custom themes", "allow custom themes"),
    ("[v2.1] crash on start", "[v2.1] crash on start"),
    ("Question marks break the lexer", "Question marks break the lexer"),
])
def test_clean_title(raw, clean):
    assert clean_title(raw) == clean


def test_clean_body_removes_template():
    body = (
        "<!-- please fill this in -->\n"
        "### Bug description\n"
        "The parser fails on unicode.\n\n"
        "**Is your feature request related to a problem?**\n"
        "- [x] I have checked the docs\n"
        "- [ ] I searched existing issues\n"
        "### Expected behavior\n"
        "It should parse."
    )
    out = clean_body(body)
    assert "Bug description" not in out
    assert "feature request" not in out
    assert "checked the docs" not in out
    assert "please fill" not in out
    assert "The parser fails on unicode." in out
    assert "It should parse." in out


def test_clean_body_removes_form_field_answers():
    # tensorflow style issue form
    body = (
        "### Issue type\n\nSupport\n\n"
        "### Have you reproduced the bug with TensorFlow Nightly?\n\nYes\n\n"
        "### TensorFlow version\n\n2.15\n\n"
        "### Current behavior?\n\n"
        "Model.fit hangs after the first epoch when using a custom generator.\n"
    )
    out = clean_body(body)
    assert "Support" not in out and "Yes" not in out and "2.15" not in out
    assert "Model.fit hangs" in out


def test_clean_body_keeps_short_real_sentences_and_code_comments():
    body = (
        "### Description\n\nIt crashes.\n\n"
        "```python\n# load the data\nimport numpy as np\n```\n"
        "Bug\n"
    )
    out = clean_body(body)
    assert "It crashes." in out
    assert "# load the data" in out
    assert "\nBug" not in out


def test_clean_body_shortens_code_and_caps_length():
    code = "```\n" + "\n".join(f"line {i}" for i in range(50)) + "\n```"
    out = clean_body("trace:\n" + code)
    assert "line 3" in out and "line 40" not in out
    assert len(clean_body("word " * 1000, max_chars=200)) <= 210


def test_stratified_split_is_disjoint_balanced_and_deterministic():
    recs = make_issues(40)
    a = stratified_split(recs, seed=1)
    b = stratified_split(recs, seed=1)
    assert [r["id"] for r in a[2]] == [r["id"] for r in b[2]]
    ids = [set(r["id"] for r in s) for s in a]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(s) for s in ids) == len(recs)
    assert set(Counter(r["label"] for r in a[2]).values()) == {8}


def test_repo_holdout_split():
    recs = make_issues(20)
    train, val, test = repo_holdout_split(recs, "acme/beta")
    assert {r["repo"] for r in test} == {"acme/beta"}
    assert {r["repo"] for r in train + val} == {"acme/alpha"}


def test_balance_caps_each_bucket():
    recs = make_issues(40)
    out = balance(recs, 5)
    counts = Counter((r["repo"], r["label"]) for r in out)
    assert set(counts.values()) == {5}


def test_to_text_has_title_and_body():
    t = to_text({"title": "T", "body": "B"})
    assert t.startswith("Title: T") and t.endswith("B")
