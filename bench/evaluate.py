"""score every saved prediction file and write the report

python -m bench.evaluate

writes results/report.md, results/summary.json and two plots
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import LABELS, read_jsonl

LABELS_PLUS = LABELS + ["invalid"]


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def bootstrap_ci(y, p, n=2000, seed=0):
    """95% interval for macro f1 by resampling test issues"""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(macro_f1(y[i], p[i]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def paired_diff(y, pa, pb, n=2000, seed=0):
    """is method a really better than b, same resamples for both"""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def score(rows):
    y = [r["label"] for r in rows]
    p = [r["pred"] for r in rows]
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    lo, hi = bootstrap_ci(y, p)
    lat = np.array([r["latency_s"] for r in rows])
    cost = sum(r["cost_usd"] for r in rows)
    n = len(rows)
    return {
        "n": n,
        "accuracy": accuracy_score(y, p),
        "macro_f1": macro_f1(y, p),
        "macro_f1_ci": [lo, hi],
        "per_class": {l: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
                      for i, l in enumerate(LABELS)},
        "confusion": confusion_matrix(y, p, labels=LABELS_PLUS).tolist(),
        "invalid": sum(x == "invalid" for x in p),
        "cost_usd": cost,
        "cost_per_1k": cost / n * 1000 if n else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50) * 1000),
        "latency_p95_ms": float(np.percentile(lat, 95) * 1000),
        "tokens_in_avg": float(np.mean([r["input_tokens"] for r in rows])),
    }


def load_all(results):
    out = {}
    for f in sorted((Path(results) / "preds").glob("*.jsonl")):
        meta_f = f.with_suffix(".meta.json")
        out[f.stem] = (read_jsonl(f), json.load(open(meta_f)) if meta_f.exists() else {})
    return out


def write_errors(results, name, rows, issues, n=12, seed=0):
    """sample of mistakes, read these before writing any conclusions"""
    wrong = [r for r in rows if r["pred"] != r["label"]]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(wrong), min(n, len(wrong)), replace=False) if wrong else []
    lines = [f"# errors: {name}", "", f"{len(wrong)} of {len(rows)} wrong, showing {len(pick)}", ""]
    for i in pick:
        r = wrong[i]
        rec = issues.get(r["id"], {})
        body = (rec.get("body") or "")[:400].replace("\n", " ")
        lines += [f"## {r['id']}  true={r['label']}  pred={r['pred']}",
                  f"{rec.get('url', '')}", "", f"**{rec.get('title', '')}**", "", f"> {body}", ""]
    path = Path(results) / "errors" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plots(results, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    # cost vs quality, free methods sit on the left edge
    fig, ax = plt.subplots(figsize=(8, 4.8))
    floor = 0.01
    markers = "osD^vP*X"
    for i, nm in enumerate(names):
        s = summary[nm]
        x = max(s["cost_per_1k"], floor)
        lo, hi = s["macro_f1_ci"]
        ax.errorbar(x, s["macro_f1"], yerr=[[s["macro_f1"] - lo], [hi - s["macro_f1"]]],
                    fmt=markers[i % len(markers)], capsize=4, ms=7, label=nm)
    # legend outside so labels never collide with points
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel("cost per 1,000 issues (USD, log scale; free methods drawn at $0.01)")
    ax.set_ylabel("macro F1 (95% bootstrap CI)")
    ax.set_title("Issue triage: quality vs cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(results) / "f1_vs_cost.png", dpi=150)
    plt.close(fig)

    # confusion matrices side by side
    k = len(names)
    cols = min(k, 3)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax, nm in zip(axes.ravel(), names):
        cm = np.array(summary[nm]["confusion"])[:3]
        show = cm[:, :3] if cm[:, 3].sum() == 0 else cm
        xl = LABELS if show.shape[1] == 3 else LABELS_PLUS
        norm = show / np.maximum(show.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, show[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(xl)), xl, fontsize=8)
        ax.set_yticks(range(3), LABELS, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        ax.set_title(nm, fontsize=9)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(results) / "confusion.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--baseline", default="tfidf-logreg")
    args = p.parse_args(argv)

    runs = load_all(args.results)
    if not runs:
        raise SystemExit("no prediction files in results/preds, run bench.run first")

    # every method must be scored on the same issues or the comparison is meaningless
    id_sets = {nm: {r["id"] for r in rows} for nm, (rows, _) in runs.items()}
    common = set.intersection(*id_sets.values())
    for nm, ids in id_sets.items():
        if ids != common:
            print(f"note: {nm} has {len(ids)} issues, scoring all methods on the {len(common)} they share")

    issues = {}
    issues_f = Path(args.data) / "issues.jsonl"
    if issues_f.exists():
        issues = {r["id"]: r for r in read_jsonl(issues_f)}

    summary, aligned = {}, {}
    for nm, (rows, meta) in runs.items():
        rows = sorted([r for r in rows if r["id"] in common], key=lambda r: r["id"])
        aligned[nm] = rows
        summary[nm] = score(rows)
        summary[nm]["meta"] = {k: v for k, v in meta.items() if k not in ("top_features", "history")}
        write_errors(args.results, nm, rows, issues)

    order = sorted(summary, key=lambda nm: -summary[nm]["macro_f1"])
    summary = {nm: summary[nm] for nm in order}

    base = args.baseline if args.baseline in aligned else None
    if base:
        y = [r["label"] for r in aligned[base]]
        pb = [r["pred"] for r in aligned[base]]
        for nm in order:
            if nm == base:
                continue
            pa = [r["pred"] for r in aligned[nm]]
            summary[nm]["vs_baseline"] = dict(zip(["mean", "lo", "hi"], paired_diff(y, pa, pb)))

    json.dump(summary, open(Path(args.results) / "summary.json", "w"), indent=2)
    plots(args.results, summary)

    lines = ["# Results", "", f"{len(common)} test issues.", "",
             "| method | macro F1 (95% CI) | accuracy | bug F1 | feature F1 | question F1 | cost / 1k issues | p50 latency |",
             "|---|---|---|---|---|---|---|---|"]
    for nm in order:
        s = summary[nm]
        pc = s["per_class"]
        lo, hi = s["macro_f1_ci"]
        lines.append(
            f"| {nm} | {s['macro_f1']:.3f} ({lo:.3f}–{hi:.3f}) | {s['accuracy']:.3f} | "
            f"{pc['bug']['f1']:.3f} | {pc['feature']['f1']:.3f} | {pc['question']['f1']:.3f} | "
            f"${s['cost_per_1k']:.2f} | {s['latency_p50_ms']:.1f} ms |")
    if base:
        lines += ["", f"Paired bootstrap difference in macro F1 vs `{base}` "
                      "(an interval that excludes 0 means the difference is real):", ""]
        for nm in order:
            if "vs_baseline" in summary[nm]:
                d = summary[nm]["vs_baseline"]
                flag = "yes" if d["lo"] > 0 or d["hi"] < 0 else "no"
                lines.append(f"- `{nm}`: {d['mean']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f}), significant: {flag}")
    inval = {nm: s["invalid"] for nm, s in summary.items() if s["invalid"]}
    if inval:
        lines += ["", "Unparseable LLM answers (counted as wrong): " + ", ".join(f"{k}={v}" for k, v in inval.items())]
    tf = runs.get(base, (None, {}))[1].get("top_features") if base else None
    if tf:
        lines += ["", "Top TF-IDF word features per class (check these for label leakage):", ""]
        for cls, feats in tf.items():
            lines.append(f"- **{cls}**: {', '.join(feats)}")
    lines += ["", "![quality vs cost](f1_vs_cost.png)", "", "![confusion matrices](confusion.png)", ""]
    (Path(args.results) / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6 + len(order)]))
    print(f"\nwrote {args.results}/report.md")


if __name__ == "__main__":
    main()
