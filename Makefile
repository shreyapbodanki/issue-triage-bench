PY ?= python
HAIKU = claude-haiku-5-5
SONNET = claude-sonnet-5-5

.PHONY: check-labels data trial all baselines claude report test clean-results

check-labels:
	$(PY) -m bench.build_dataset --check-labels

data:
	$(PY) -m bench.build_dataset

# cheap sanity check on 20 issues before spending real money
trial:
	$(PY) -m bench.run claude --model $(HAIKU) --limit 20 --name trial-haiku-zeroshot
	$(PY) -m bench.run claude --model $(HAIKU) --shots 3 --select retrieval --limit 20 --name trial-haiku-3shot

baselines:
	$(PY) -m bench.run tfidf
	$(PY) -m bench.run distilbert

claude:
	$(PY) -m bench.run claude --model $(HAIKU) --shots 0
	$(PY) -m bench.run claude --model $(HAIKU) --shots 3 --select random
	$(PY) -m bench.run claude --model $(HAIKU) --shots 3 --select retrieval
	$(PY) -m bench.run claude --model $(SONNET) --shots 0 --max-cost 3
	$(PY) -m bench.run claude --model $(SONNET) --shots 3 --select retrieval --max-cost 5

report:
	rm -f results/preds/trial-*
	$(PY) -m bench.evaluate

all: baselines claude report

test:
	$(PY) -m pytest -q

clean-results:
	rm -rf results/preds results/errors results/*.png results/*.md results/summary.json
