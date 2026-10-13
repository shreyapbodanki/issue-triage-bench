# issue-triage-bench

Do you need an LLM to triage GitHub issues? This benchmark sorts 2,700 real issues from six popular projects into **bug**, **feature**, or **question**, and compares a classic ML baseline, a fine-tuned transformer, and Claude, on quality, cost, and latency.

## Results

<!-- paste the table from results/report.md here -->


## Methods

- **TF-IDF + logistic regression**: a classic baseline that trains in seconds and is free to run
- **DistilBERT**: fine-tuned on the training split
- **Claude zero-shot**: instructions only
- **Claude few-shot**: labeled examples, either random or retrieved from the most similar training issues

## Dataset

Closed issues from pandas, NumPy, React, TensorFlow, Kubernetes, and VS Code, with maintainer labels as ground truth. The dataset is balanced to 900 issues per class and split 60/20/20.

- **Leakage removed:** issue templates give the answer away, through title tags like `[DevTools Bug]` and form fields like `Issue type: Support`. These are stripped before training. Removing them lowered baseline accuracy from 0.817 to 0.800, which is the honest number.
- **Ambiguous issues dropped:** issues labeled with two classes are excluded.
- **Validation-only tuning:** the test set is never used for tuning.

Only issue IDs are committed (`data/split_ids.json`). The issue text stays on GitHub.

## Evaluation

The headline metric is macro F1, with 95% bootstrap confidence intervals. Methods are compared with a paired bootstrap, so a reported gap is real rather than noise. Misclassified examples for each method are in `results/errors/`.

## Run it

```bash
pip install -r requirements.txt -r requirements-bert.txt
export GITHUB_TOKEN=...  ANTHROPIC_API_KEY=...

make data    # fetch, clean, split
make trial   # 20-issue sanity check, a few cents
make all     # every method, then results/report.md
pytest -q
```

Claude runs print a cost estimate first and stop above `--max-cost`. API responses are cached, so reruns are free.

## License

MIT
