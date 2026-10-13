"""tiny synthetic issues for tests only, never used for results"""
import random
from types import SimpleNamespace

from bench.data import stratified_split, write_jsonl

_BUG = [
    ("{f} crashes with TypeError when input is empty", "Calling {f} on an empty frame raises TypeError: unsupported operand. Expected an empty result. Traceback below."),
    ("Regression in {v}: {f} returns wrong values", "After upgrading to {v} the output of {f} is off by one. This worked in the previous release."),
    ("Segfault in {f} on large arrays", "Running {f} with 10 million rows segfaults. Stack trace attached, reproducible every time."),
    ("{f} ignores the dtype argument", "Passing dtype has no effect, the result is always float64 which contradicts the docs."),
]
_FEAT = [
    ("Add support for {x} in {f}", "It would be great if {f} could accept {x}. Right now we have to convert manually. Proposal: add a keyword."),
    ("Allow {f} to take a callback", "Feature idea: let users pass a function so {f} can be customized. Other libraries offer this."),
    ("Option to disable warnings in {f}", "Please add a flag to silence the warning, it would make scripts cleaner."),
    ("New method similar to {f} for {x}", "Would you consider adding a method like {f} that works on {x}? Happy to open a PR."),
]
_QST = [
    ("How do I use {f} with {x}?", "I am new to the library and cannot figure out how to call {f} with {x}. What is the right way to do this?"),
    ("What is the difference between {f} and apply?", "Trying to understand when I should use {f} instead of apply. The docs were not clear to me."),
    ("Is it possible to configure {f} globally?", "Is there a setting so I do not have to pass the option every time? Not sure where to look."),
    ("Help installing {v} on Windows", "How can I install {v} on Windows, pip gives me a message I do not understand. Any advice?"),
]
_F = ["groupby", "merge", "read_csv", "concat", "pivot", "resample", "rolling", "to_json"]
_X = ["datetime columns", "nested lists", "sparse data", "categoricals", "arrow tables"]
_V = ["2.1", "2.2", "3.0", "1.26"]


def make_issues(n_per_class=40, seed=0):
    rng = random.Random(seed)
    out = []
    for label, temps in [("bug", _BUG), ("feature", _FEAT), ("question", _QST)]:
        for i in range(n_per_class):
            t, b = rng.choice(temps)
            kw = {"f": rng.choice(_F), "x": rng.choice(_X), "v": rng.choice(_V)}
            repo = "acme/alpha" if i % 2 else "acme/beta"
            out.append({"id": f"{repo}#{label}{i}", "repo": repo, "title": t.format(**kw),
                        "body": b.format(**kw), "label": label, "url": ""})
    return out


def write_fixture(data_dir):
    recs = make_issues()
    train, val, test = stratified_split(recs)
    write_jsonl(f"{data_dir}/issues.jsonl", recs)
    for name, s in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(f"{data_dir}/splits/{name}.jsonl", s)
    return train, val, test


class FakeClient:
    """stands in for anthropic.Anthropic, answers by keyword"""

    def __init__(self, fail_first=0):
        self.calls = 0
        self.fail_first = fail_first
        self.messages = self

    def create(self, model, max_tokens, temperature, system, messages):
        self.calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            err = Exception("overloaded")
            err.status_code = 529
            raise err
        text = messages[-1]["content"].lower()
        if any(w in text for w in ("how ", "what ", "?", "help")):
            ans = "question"
        elif any(w in text for w in ("add", "allow", "option", "feature", "new method")):
            ans = "feature"
        else:
            ans = "Bug"
        chars = len(system) + sum(len(m["content"]) for m in messages)
        return SimpleNamespace(content=[SimpleNamespace(text=ans)],
                               usage=SimpleNamespace(input_tokens=chars // 4, output_tokens=1))
