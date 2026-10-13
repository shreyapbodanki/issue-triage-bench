"""fine tune a small transformer on the train split

runs on cpu, apple silicon (mps) or cuda, whichever is available
distilbert is 66m parameters so a few epochs on ~1500 issues takes
roughly 10 to 30 minutes on a laptop cpu and a couple of minutes on a gpu
"""
import time

import numpy as np

from ..data import LABELS, to_text


def _device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batches(enc, labels, bs, shuffle, rng):
    idx = np.arange(len(labels))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield {k: v[j] for k, v in enc.items()}, labels[j]


def _predict(torch, model, enc, bs, device):
    model.eval()
    n = enc["input_ids"].shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = {k: v[i:i + bs].to(device) for k, v in enc.items()}
            preds.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def train_and_predict(train, val, test, model=None, tokenizer=None, model_name="distilbert-base-uncased",
                      epochs=4, lr=5e-5, batch_size=16, max_len=256, seed=42, device=None, **_):
    import torch
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if tokenizer is None or model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    device = device or _device(torch)
    model.to(device)

    def encode(rows):
        return dict(tokenizer([to_text(r) for r in rows], truncation=True, max_length=max_len,
                              padding="max_length", return_tensors="pt"))

    lab = {l: i for i, l in enumerate(LABELS)}
    enc_tr, enc_val, enc_te = encode(train), encode(val), encode(test)
    y_tr = torch.tensor([lab[r["label"]] for r in train])
    y_val = np.array([lab[r["label"]] for r in val])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, (steps - s) / steps))

    t0 = time.perf_counter()
    best = (-1.0, None, -1)
    history = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in _batches(enc_tr, y_tr, batch_size, True, rng):
            xb = {k: v.to(device) for k, v in xb.items()}
            loss = model(**xb, labels=yb.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        # keep the epoch with the best val macro f1
        vf1 = f1_score(y_val, _predict(torch, model, enc_val, 64, device), average="macro") if len(val) else 0.0
        history.append({"epoch": ep + 1, "train_loss": total, "val_macro_f1": float(vf1)})
        print(f"  epoch {ep + 1}  loss {total:.3f}  val macro f1 {vf1:.3f}")
        if vf1 > best[0]:
            best = (vf1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
    train_s = time.perf_counter() - t0
    model.load_state_dict(best[1])

    t0 = time.perf_counter()
    pred = _predict(torch, model, enc_te, 64, device)
    per_item = (time.perf_counter() - t0) / max(len(test), 1)

    rows = [{"id": r["id"], "label": r["label"], "pred": LABELS[p], "latency_s": per_item,
             "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            for r, p in zip(test, pred)]
    meta = {"model": model_name, "device": device, "best_epoch": best[2], "val_macro_f1": best[0],
            "train_seconds": train_s, "history": history,
            "hparams": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "max_len": max_len}}
    return rows, meta
