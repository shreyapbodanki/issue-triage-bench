"""runs only where torch and transformers are installed

uses a tiny randomly initialized model and a tokenizer built on the fly,
so it checks the training loop without downloading anything
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from bench.data import LABELS, stratified_split, to_text  # noqa: E402
from bench.methods import distilbert  # noqa: E402
from fixtures import make_issues  # noqa: E402


def _tiny(train):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([to_text(r).lower() for r in train],
                            trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    cfg = DistilBertConfig(vocab_size=fast.vocab_size, dim=32, hidden_dim=64, n_layers=2, n_heads=2,
                           num_labels=len(LABELS), max_position_embeddings=128)
    return DistilBertForSequenceClassification(cfg), fast


def test_tiny_model_trains_and_predicts():
    train, val, test = stratified_split(make_issues(30))
    model, tok = _tiny(train)
    rows, meta = distilbert.train_and_predict(train, val, test, model=model, tokenizer=tok,
                                              epochs=8, lr=1e-3, batch_size=8, max_len=64, device="cpu")
    assert len(rows) == len(test)
    assert all(r["pred"] in LABELS for r in rows)
    acc = sum(r["pred"] == r["label"] for r in rows) / len(rows)
    assert acc > 0.5  # synthetic data is easy, random would be about 0.33
    assert meta["best_epoch"] >= 1
