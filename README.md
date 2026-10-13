# issue-triage-bench

**Do you need an LLM to triage GitHub issues?** This repo benchmarks four ways of sorting real issues from popular open source projects into **bug**, **feature request**, or **question**:

| approach | what it is |
|---|---|
| TF-IDF + logistic regression | classic ML baseline, trains in seconds, free to run |
| Fine-tuned DistilBERT | 66M-parameter transformer fine-tuned on the training split |
| Claude zero-shot | instructions only, no examples |
| Claude few-shot | instructions plus labeled examples, either a fixed random set or **retrieved** nearest neighbors from the training split |

Every method is scored on the same held-out issues for **quality** (macro F1 with bootstrap confidence intervals), **cost** per 1,000 issues, and **latency**.

## Results

> Run the benchmark (below), then paste `results/report.md` here and embed the plots.

![quality vs cost](results/f1_vs_cost.png)

## The dataset

Closed issues from projects whose maintainers label issues consistently (pandas, NumPy, React, TensorFlow, Kubernetes, VS Code; see `config/repos.json`). The maintainers' labels are the ground truth.

Building a fair dataset took more care than training the models:

- **Label leakage is removed.** Issue templates insert headings like `### Bug description`, and some projects prefix titles with `BUG:` or `ENH:`. Left in, these make the task trivial for every method. `bench/data.py` strips template headings, checkboxes, HTML comments, and title prefixes, keeping only what the author actually wrote. The report lists the TF-IDF model's most predictive words per class, so any leftover leakage is visible.
- **Ambiguous issues are dropped.** Issues carrying labels from two classes are excluded rather than resolved arbitrarily.
- **Each project is balanced.** Each (repo, label) bucket is capped, so one large project can't dominate.
- **Hyperparameters are tuned on validation only.** The regularization strength, the best fine-tuning epoch, and the prompt are all chosen without touching the test set.
- **Cross-project generalization is optional.** `--holdout-repo` trains on five projects and tests on the sixth. This shows whether a method learned what a bug *is*, or only memorized one project's vocabulary.

Only issue IDs are committed (`data/split_ids.json`); the text stays on GitHub.

## Run it

```bash
pip install -r requirements.txt
pip install -r requirements-bert.txt        # only for the DistilBERT run

export GITHUB_TOKEN=$(gh auth token)        # or any token, no scopes needed
export ANTHROPIC_API_KEY=sk-ant-...

make check-labels   # confirm label names in config/repos.json exist
make data           # fetch, clean, balance, split
make trial          # 20 issues per Claude config, a few cents, to sanity check
make all            # every method, then results/report.md
```

Or run methods one at a time:

```bash
python -m bench.run tfidf
python -m bench.run distilbert --epochs 4
python -m bench.run claude --model claude-haiku-5-5 --shots 0
python -m bench.run claude --model claude-haiku-5-5 --shots 3 --select retrieval
python -m bench.evaluate
```

**Cost control:** every Claude run prints an estimate first and stops if it exceeds `--max-cost` (default $1). Responses are cached in `results/cache/`, so reruns and report tweaks cost nothing. Prices are in `config/prices.json`; check them against the [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) before quoting costs.

## How the evaluation works

- **Macro F1** is the headline metric because it weights the three classes equally. Accuracy is reported too.
- **95% confidence intervals** come from 2,000 bootstrap resamples of the test set. With a few hundred test issues, a difference of 0.02 F1 is often noise, and the intervals make that visible.
- **Paired bootstrap vs. the baseline** applies the same resamples to both methods, which is the right test for "is method A actually better than B on these issues."
- **Unparseable LLM answers** count as wrong and are reported separately.
- **Error samples** for every method are written to `results/errors/`. Read them before drawing conclusions: some "errors" are really mislabeled issues.

## Layout

```
bench/build_dataset.py   GitHub API fetch, cleaning, balancing, splits
bench/data.py            cleaning rules and split logic
bench/methods/tfidf.py   baseline
bench/methods/distilbert.py  fine-tuning loop (CPU, MPS, or CUDA)
bench/methods/llm.py     Claude classifier, few-shot selection, cache, retries
bench/run.py             run one method, save predictions
bench/evaluate.py        metrics, bootstrap CIs, report, plots
tests/                   unit tests and an end-to-end run with a fake API client
```

```bash
pytest -q
```

## License

MIT for the code. Issue text belongs to its authors and is fetched from GitHub at build time.
