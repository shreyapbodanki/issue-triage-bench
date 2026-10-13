"""claude as a classifier, zero shot and few shot

few shot examples come from the train split only, picked either at random
or by retrieval, meaning the training issues most similar to the one being classified
every api response is cached on disk so reruns and report tweaks cost nothing
"""
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..data import LABELS, to_text

SYSTEM = """You are triaging GitHub issues for an open source project. Classify the issue into exactly one category:

bug: something is broken. An error, crash, wrong result, regression, or behavior that contradicts the documentation.
feature: a request for new functionality or a change to how something works. Nothing is broken.
question: the author needs help using the project, setting it up, or understanding its behavior. It is not reporting a defect or asking for a change.

Reply with only one word: bug, feature, or question."""

_LABEL_RE = re.compile(r"\b(bug|feature|question)\b")
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


def parse_label(text):
    m = _LABEL_RE.search((text or "").lower())
    return m.group(1) if m else "invalid"


def fmt_issue(text, max_chars=None):
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return f"<issue>\n{text}\n</issue>"


class FewShotSelector:
    def __init__(self, train, k_per_class, mode="random", seed=42):
        self.k = k_per_class
        self.mode = mode
        self.by_label = {l: [r for r in train if r["label"] == l] for l in LABELS}
        rng = random.Random(seed)
        # one fixed random set so every test issue sees the same examples
        self.fixed = {l: rng.sample(v, min(self.k, len(v))) for l, v in self.by_label.items()}
        if mode == "retrieval":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self.train = train
            self.X = self.vec.fit_transform([to_text(r) for r in train])

    def pick(self, rec):
        if self.k == 0:
            return []
        if self.mode == "random":
            chosen = self.fixed
        else:
            sims = (self.X @ self.vec.transform([to_text(rec)]).T).toarray().ravel()
            order = sims.argsort()[::-1]
            chosen = {l: [] for l in LABELS}
            for i in order:
                r = self.train[i]
                if r["id"] != rec["id"] and len(chosen[r["label"]]) < self.k:
                    chosen[r["label"]].append(r)
                if all(len(v) >= self.k for v in chosen.values()):
                    break
        # round robin across classes so the last example is not always the same label
        out = []
        for i in range(self.k):
            for l in LABELS:
                if i < len(chosen[l]):
                    out.append(chosen[l][i])
        return out


class DiskCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.data = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    self.data[row["key"]] = row

    def get(self, key):
        return self.data.get(key)

    def put(self, row):
        with self.lock:
            self.data[row["key"]] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


class ClaudeClassifier:
    def __init__(self, model, prices, shots=0, select="random", train=None,
                 client=None, cache_dir="results/cache", workers=4, example_chars=700, seed=42):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=0)  # we retry ourselves
        self.client = client
        self.model = model
        self.price = prices.get(model)
        if self.price is None:
            raise ValueError(f"no price for {model} in config/prices.json")
        self.selector = FewShotSelector(train or [], shots, select, seed)
        self.example_chars = example_chars
        self.cache = DiskCache(Path(cache_dir) / f"{model}.jsonl")
        self.workers = workers

    def build_messages(self, rec):
        msgs = []
        for ex in self.selector.pick(rec):
            msgs.append({"role": "user", "content": fmt_issue(to_text(ex), self.example_chars)})
            msgs.append({"role": "assistant", "content": ex["label"]})
        msgs.append({"role": "user", "content": fmt_issue(to_text(rec))})
        return msgs

    def cost(self, tin, tout):
        return tin / 1e6 * self.price["input"] + tout / 1e6 * self.price["output"]

    def estimate_cost(self, records):
        # rough, about 3.5 characters per token for english plus code
        chars = sum(len(SYSTEM) + sum(len(m["content"]) for m in self.build_messages(r)) for r in records)
        tin = chars / 3.5
        return self.cost(tin, 3 * len(records))

    def _call(self, msgs):
        key = hashlib.sha256(json.dumps([self.model, SYSTEM, msgs], sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit:
            return hit, True
        for attempt in range(7):
            try:
                t0 = time.perf_counter()
                resp = self.client.messages.create(
                    model=self.model, max_tokens=5, temperature=0,
                    system=SYSTEM, messages=msgs,
                )
                latency = time.perf_counter() - t0
                break
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                name = type(e).__name__
                if status in RETRYABLE or name in ("APIConnectionError", "APITimeoutError"):
                    time.sleep(min(2 ** attempt + random.random(), 60))
                    continue
                raise
        else:
            raise RuntimeError("api kept failing, try fewer --workers")
        text = "".join(getattr(b, "text", "") for b in resp.content)
        row = {"key": key, "text": text, "latency_s": latency,
               "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        self.cache.put(row)
        return row, False

    def classify(self, rec):
        row, cached = self._call(self.build_messages(rec))
        return {
            "id": rec["id"], "label": rec["label"], "pred": parse_label(row["text"]),
            "raw": row["text"], "latency_s": row["latency_s"],
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
            "cost_usd": self.cost(row["input_tokens"], row["output_tokens"]), "cached": cached,
        }

    def run(self, records, progress=True):
        out = [None] * len(records)
        done = 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self.classify, r): i for i, r in enumerate(records)}
            for f in futs:
                out[futs[f]] = f.result()
                done += 1
                if progress and (done % 25 == 0 or done == len(records)):
                    spent = sum(o["cost_usd"] for o in out if o and not o["cached"])
                    print(f"  {done}/{len(records)}  new spend ${spent:.4f}")
        return out
