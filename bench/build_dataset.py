"""build the github issues dataset

python -m bench.build_dataset --config config/repos.json --out data

set GITHUB_TOKEN first, without it github allows 60 requests per hour
which is not enough for more than a couple of repos
a token with no scopes is fine since these are public repos
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

from .data import (LABELS, balance, clean_body, clean_title, read_jsonl, repo_holdout_split,
                   stratified_split, write_jsonl)

API = "https://api.github.com"


class GitHub:
    def __init__(self, token=None):
        self.s = requests.Session()
        self.s.headers["Accept"] = "application/vnd.github+json"
        self.s.headers["X-GitHub-Api-Version"] = "2022-11-28"
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"

    def get(self, path, params=None):
        for attempt in range(5):
            r = self.s.get(API + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            remaining = r.headers.get("X-RateLimit-Remaining")
            if r.status_code in (403, 429) and remaining == "0":
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 1) + 2
                if wait > 300:
                    sys.exit(f"rate limited for {wait / 60:.0f} more minutes, set GITHUB_TOKEN and rerun")
                print(f"  rate limited, sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"github {r.status_code} on {path}: {r.text[:200]}")
        raise RuntimeError(f"gave up on {path}")

    def labels(self, repo):
        names, page = set(), 1
        while True:
            batch = self.get(f"/repos/{repo}/labels", {"per_page": 100, "page": page})
            names |= {x["name"] for x in batch}
            if len(batch) < 100:
                return names
            page += 1

    def issues(self, repo, label, want, max_pages):
        """closed issues with this label, pull requests filtered out"""
        out, page = [], 1
        while len(out) < want and page <= max_pages:
            batch = self.get(f"/repos/{repo}/issues", {
                "labels": label, "state": "closed", "per_page": 100, "page": page,
                "sort": "created", "direction": "desc",
            })
            out += [x for x in batch if "pull_request" not in x]
            if len(batch) < 100:
                break
            page += 1
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/repos.json")
    p.add_argument("--out", default="data")
    p.add_argument("--per-class", type=int, default=150, help="max issues per label per repo")
    p.add_argument("--max-pages", type=int, default=4)
    p.add_argument("--min-chars", type=int, default=40, help="drop issues with almost no text")
    p.add_argument("--holdout-repo", default=None, help="test on this repo only, train on the rest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-labels", action="store_true", help="only verify label names exist")
    p.add_argument("--from-raw", action="store_true",
                   help="reuse issues saved in data/raw by an earlier run, no github requests")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    raw_dir = Path(args.out) / "raw"
    gh = None
    if not args.from_raw:
        gh = GitHub(os.environ.get("GITHUB_TOKEN"))
        if not os.environ.get("GITHUB_TOKEN"):
            print("warning: no GITHUB_TOKEN, you will hit the 60 per hour limit fast")

        # make sure every configured label exists before spending requests
        for entry in cfg["repos"]:
            existing = gh.labels(entry["repo"])
            for cls, names in entry["labels"].items():
                missing = [n for n in names if n not in existing]
                if missing:
                    print(f"  {entry['repo']}: label(s) {missing} for {cls} not found")
        if args.check_labels:
            return

    def fetch(repo, name):
        # raw api results are saved so cleaning changes never need a refetch
        f = raw_dir / f"{repo.replace('/', '__')}__{re.sub(r'[^A-Za-z0-9]+', '_', name)}.jsonl"
        if args.from_raw:
            if not f.exists():
                sys.exit(f"no raw file {f}, run once without --from-raw")
            return read_jsonl(f)
        items = gh.issues(repo, name, args.per_class * 2, args.max_pages)
        slim = [{"number": x["number"], "title": x["title"], "body": x.get("body") or "",
                 "labels": [{"name": l["name"]} for l in x["labels"]],
                 "created_at": x["created_at"], "html_url": x["html_url"]} for x in items]
        write_jsonl(f, slim)
        return slim

    raw, records = [], {}
    for entry in cfg["repos"]:
        repo = entry["repo"]
        # label name -> class, used to spot issues tagged as two classes
        name_to_cls = {n: c for c, names in entry["labels"].items() for n in names}
        for cls, names in entry["labels"].items():
            assert cls in LABELS, f"unknown class {cls}"
            for name in names:
                items = fetch(repo, name)
                print(f"  {repo:30s} {name:22s} -> {cls:8s} {len(items)} issues")
                for it in items:
                    classes = {name_to_cls[l["name"]] for l in it["labels"] if l["name"] in name_to_cls}
                    raw.append({"repo": repo, "number": it["number"], "classes": sorted(classes)})
                    if len(classes) != 1:
                        continue  # ambiguous, labeled as two classes
                    rec = {
                        "id": f"{repo}#{it['number']}",
                        "repo": repo,
                        "title": clean_title(it["title"]),
                        "body": clean_body(it.get("body") or ""),
                        "label": cls,
                        "created_at": it["created_at"],
                        "url": it["html_url"],
                    }
                    if len(rec["title"]) + len(rec["body"]) >= args.min_chars:
                        records[rec["id"]] = rec

    recs = balance(list(records.values()), args.per_class, args.seed)
    ambiguous = len({(r["repo"], r["number"]) for r in raw if len(r["classes"]) > 1})
    print(f"\n{len(records)} usable issues, {ambiguous} skipped for conflicting labels")
    print(f"{len(recs)} after balancing to {args.per_class} per label per repo")

    if args.holdout_repo:
        train, val, test = repo_holdout_split(recs, args.holdout_repo, seed=args.seed)
    else:
        train, val, test = stratified_split(recs, seed=args.seed)

    write_jsonl(f"{args.out}/issues.jsonl", recs)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(f"{args.out}/splits/{name}.jsonl", split)
        counts = Counter(r["label"] for r in split)
        print(f"  {name:5s} {len(split):4d}  " + "  ".join(f"{l}={counts[l]}" for l in LABELS))

    meta = {"config": cfg, "args": vars(args), "n": len(recs),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    # ids only, so the split can be published without rehosting issue text
    ids = {name: [r["id"] for r in s] for name, s in [("train", train), ("val", val), ("test", test)]}
    json.dump(ids, open(f"{args.out}/split_ids.json", "w"), indent=1)


if __name__ == "__main__":
    main()
