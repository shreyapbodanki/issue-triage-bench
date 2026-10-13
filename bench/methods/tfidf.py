"""classic baseline, tf-idf features into logistic regression

cheap, fast and a surprisingly hard bar for llms to clear
"""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

from ..data import to_text


def _features():
    # words plus character ngrams, char ngrams catch things like "TypeError" and "v2.1"
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)),
    ])


def run(train, val, test, Cs=(0.3, 1.0, 3.0, 10.0), **_):
    xtr = [to_text(r) for r in train]
    ytr = [r["label"] for r in train]

    t0 = time.perf_counter()
    feats = _features()
    Xtr = feats.fit_transform(xtr)
    Xval = feats.transform([to_text(r) for r in val])
    yval = [r["label"] for r in val]

    # pick regularization on val so test stays untouched
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        score = f1_score(yval, clf.predict(Xval), average="macro") if val else 0.0
        if best is None or score > best[0]:
            best = (score, C, clf)
    val_f1, C, clf = best
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = clf.predict(feats.transform([to_text(r) for r in test]))
    per_item = (time.perf_counter() - t0) / max(len(test), 1)

    rows = [{"id": r["id"], "label": r["label"], "pred": p, "latency_s": per_item,
             "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            for r, p in zip(test, preds)]
    meta = {"C": C, "val_macro_f1": val_f1, "train_seconds": train_s,
            "top_features": top_features(feats, clf)}
    return rows, meta


def top_features(feats, clf, n=15):
    """most predictive word features per class, a quick check for label leakage"""
    names = np.array(feats.transformer_list[0][1].get_feature_names_out())
    n_word = len(names)
    out = {}
    for i, cls in enumerate(clf.classes_):
        coef = clf.coef_[i][:n_word] if len(clf.classes_) > 2 else clf.coef_[0][:n_word]
        out[cls] = names[np.argsort(coef)[::-1][:n]].tolist()
    return out
