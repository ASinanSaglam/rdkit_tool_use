# RDKit tool-use QA — reproducible pipeline.
# Local (6GB GPU): test, data, baseline smoke.  Colab (16GB): train, full eval.
.PHONY: help setup test data-dev data-full baseline eval-base eval-ft bench-hard clean

PY ?= python
# hard cap so a CUDA spike can't take down WSL; used for every local GPU run
GPU_ENV = PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

help:
	@grep -E '^[a-z-]+:.*#' $(MAKEFILE_LIST) | sed 's/:.*#/\t/'

setup:  # install pinned deps
	$(PY) -m pip install -r requirements.txt

test:  # oracle + harness self-checks (no GPU, no network)
	$(PY) -m rdkit_qa.oracle
	$(PY) -m rdkit_qa.harness

data-dev:  # tiny set for fast local smoke -> data/dev/
	$(PY) -m rdkit_qa.dataset --name dev

data-full:  # 10k drug-like mols for training -> data/full/
	$(PY) -m rdkit_qa.dataset --name full

baseline:  # base-model smoke on 2060: 5 examples, memory-capped
	$(GPU_ENV) $(PY) -m rdkit_qa.eval --name dev --limit 5 --tag base_smoke

eval-base:  # full base-model benchmark (run on Colab or a bigger GPU)
	$(PY) -m rdkit_qa.eval --name $(NAME) --tag base

eval-ft:  # fine-tuned benchmark; pass ADAPTER=path
	$(PY) -m rdkit_qa.eval --name $(NAME) --adapter $(ADAPTER) --tag ft

bench-hard:  # paraphrase/tool-restraint/multi-call generalization eval; pass ADAPTER= for ft
	$(PY) -m rdkit_qa.bench_hard --name $(NAME) --adapter $(ADAPTER) --tag $(TAG)

clean:  # remove generated data + results (keeps cached raw_bbbp.csv)
	rm -rf data/dev data/full results artifacts/models
