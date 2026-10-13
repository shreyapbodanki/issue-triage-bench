"""run one method and save its predictions

python -m bench.run tfidf
python -m bench.run distilbert
python -m bench.run claude --model claude-haiku-5-5 --shots 0
python -m bench.run claude --model claude-haiku-5-5 --shots 2 --select retrieval
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .data import load_split, write_jsonl


def main(argv=None, client=None):
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["tfidf", "distilbert", "claude"])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--limit", type=int, default=None, help="only the first n test issues, for a cheap trial")
    p.add_argument("--name", default=None)
    # claude options
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--shots", type=int, default=0, help="examples per class")
    p.add_argument("--select", choices=["random", "retrieval"], default="random")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-cost", type=float, default=1.0, help="stop if the estimate is above this many dollars")
    p.add_argument("--yes", action="store_true", help="skip the cost check")
    # distilbert options
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bert-model", default="distilbert-base-uncased")
    args = p.parse_args(argv)

    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    test = load_split(args.data, "test")
    if args.limit:
        test = test[: args.limit]

    t0 = time.perf_counter()
    if args.method == "tfidf":
        from .methods import tfidf
        name = args.name or "tfidf-logreg"
        rows, meta = tfidf.run(train, val, test)
    elif args.method == "distilbert":
        from .methods import distilbert
        name = args.name or "distilbert"
        rows, meta = distilbert.train_and_predict(train, val, test, model_name=args.bert_model, epochs=args.epochs)
    else:
        from .methods.llm import ClaudeClassifier
        prices = json.load(open(Path(__file__).resolve().parent.parent / "config" / "prices.json"))
        if args.shots == 0:
            name = args.name or f"{args.model}-zeroshot"
        else:
            name = args.name or f"{args.model}-{args.shots}shot-{args.select}"
        clf = ClaudeClassifier(args.model, prices, shots=args.shots, select=args.select, train=train,
                               client=client, cache_dir=Path(args.out) / "cache", workers=args.workers)
        est = clf.estimate_cost(test)
        print(f"{name}: {len(test)} issues, estimated cost ${est:.3f} (cached calls are free)")
        if est > args.max_cost and not args.yes:
            sys.exit(f"estimate is above --max-cost {args.max_cost}, rerun with --yes or a higher cap")
        rows = clf.run(test)
        meta = {"model": args.model, "shots_per_class": args.shots, "select": args.select,
                "invalid": sum(r["pred"] == "invalid" for r in rows),
                "new_spend_usd": sum(r["cost_usd"] for r in rows if not r["cached"])}

    meta.update({"method": name, "n_test": len(test), "limit": args.limit,
                 "wall_seconds": time.perf_counter() - t0})
    out = Path(args.out) / "preds"
    write_jsonl(out / f"{name}.jsonl", rows)
    json.dump(meta, open(out / f"{name}.meta.json", "w"), indent=2)
    acc = sum(r["pred"] == r["label"] for r in rows) / max(len(rows), 1)
    print(f"{name}: accuracy {acc:.3f} on {len(rows)} issues -> {out / name}.jsonl")
    return name


if __name__ == "__main__":
    main()
