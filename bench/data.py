"""loading, cleaning and splitting issues

every record looks like
{"id": "pandas-dev/pandas#123", "repo": ..., "title": ..., "body": ..., "label": "bug"}
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

LABELS = ["bug", "feature", "question"]

# title prefixes that give the answer away like "BUG:" "[ENH]" "QST -"
# only strip when bracketed or followed by : or - so "Question marks break x" survives
_WORDS = (r"(?:bug|bugs|enh|enhancement|feat|feature(?: request)?|fr|qst|question|q|"
          r"support|help|usage question|request)")
_PREFIX = re.compile(
    rf"^\s*(?:[\[\(]\s*{_WORDS}\s*[\]\)]\s*[:\-–]?|{_WORDS}\s*[:\-–])\s*",
    re.IGNORECASE,
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_BOLD_LINE = re.compile(r"^\s*\*\*[^*\n]{1,80}\*\*:?\s*$", re.MULTILINE)
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\].*$", re.MULTILINE)
_SPACES = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def _shorten_code(m, keep=6):
    lines = m.group(0).splitlines()
    if len(lines) <= keep + 2:
        return m.group(0)
    return "\n".join(lines[: keep + 1] + ["...", "```"])


def clean_title(title):
    # strip repeatedly so "[BUG] ENH: x" does not slip through
    prev = None
    while prev != title:
        prev = title
        title = _PREFIX.sub("", title, count=1)
    return title.strip()


def clean_body(body, max_chars=1500):
    """remove issue template scaffolding that leaks the label

    templates add headings like "### Bug description" or "**Is your feature request
    related to a problem**" which make classification trivially easy for every
    method, so they get dropped and only what the person actually wrote stays
    """
    if not body:
        return ""
    body = body.replace("\r\n", "\n")
    body = _HTML_COMMENT.sub("", body)
    body = _CODE_BLOCK.sub(_shorten_code, body)
    body = _HEADING.sub("", body)
    body = _BOLD_LINE.sub("", body)
    body = _CHECKBOX.sub("", body)
    body = _SPACES.sub(" ", body)
    body = _NEWLINES.sub("\n\n", body).strip()
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
    return body


def to_text(rec):
    """single string every method sees, keep it identical across methods"""
    return f"Title: {rec['title']}\n\n{rec['body']}".strip()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(data_dir, name):
    return read_jsonl(Path(data_dir) / "splits" / f"{name}.jsonl")


def balance(records, per_class_per_repo, seed=42):
    """cap each (repo, label) bucket so one big repo does not dominate"""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["repo"], r["label"])].append(r)
    out = []
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        out.extend(items[:per_class_per_repo])
    return out


def stratified_split(records, fracs=(0.6, 0.2, 0.2), seed=42):
    """split inside each label so class balance matches across splits"""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    train, val, test = [], [], []
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda r: r["id"])
        rng.shuffle(items)
        n = len(items)
        a = int(round(n * fracs[0]))
        b = a + int(round(n * fracs[1]))
        train += items[:a]
        val += items[a:b]
        test += items[b:]
    for s in (train, val, test):
        rng.shuffle(s)
    return train, val, test


def repo_holdout_split(records, holdout_repo, val_frac=0.2, seed=42):
    """train on every other repo and test on one unseen repo

    this measures whether a method learned what a bug is or just
    memorized one project's vocabulary
    """
    test = [r for r in records if r["repo"] == holdout_repo]
    rest = [r for r in records if r["repo"] != holdout_repo]
    if not test:
        raise ValueError(f"no records for holdout repo {holdout_repo}")
    train, val, _ = stratified_split(rest, (1 - val_frac, val_frac, 0.0), seed)
    return train, val, test
