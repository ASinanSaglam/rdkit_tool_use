# RDKit tool-use QA — reproducible pipeline.
# Local (6GB GPU): test, data, baseline smoke.  Colab (16GB): train, full eval.
.PHONY: help setup test data-dev data-full baseline eval-base eval-ft bench-hard clean test-v2 data-dev-v2 data-full-v2 \
	test-v3 data-dev-v3 data-full-v3 train-v3 chat-v3 eval-v3 train-v3-grpo

PY ?= python
# hard cap so a CUDA spike can't take down WSL; used for every local GPU run
GPU_ENV = PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# v3 GRPO needs trl>=1.7 (native multi-turn tools= support); def is pinned to
# trl==0.24.0 which can't run it at all -- see requirements.txt / README.
PY_GRPO ?= micromamba run -n rdkit_v3 python

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

test-v2:  # v2 (native tool-calling) self-checks -- no GPU, no network
	$(PY) -m rdkit_qa_v2.harness

data-dev-v2:  # tiny v2 set -> data/v2/dev/
	$(PY) -m rdkit_qa_v2.dataset --name dev

data-full-v2:  # full v2 set (5 behaviors) -> data/v2/full/
	$(PY) -m rdkit_qa_v2.dataset --name full

test-v3:  # v3 (name_to_smiles + rdkit_compute) self-checks -- no GPU; name_dict_pubchem demo needs network
	$(PY) -m rdkit_qa_v3.oracle
	$(PY) -m rdkit_qa_v3.harness

data-dev-v3:  # tiny v3 set -> data/v3/dev/
	$(PY) -m rdkit_qa_v3.dataset --name dev

data-full-v3:  # full v3 set (6 behaviors incl. chain) -> data/v3/full/
	$(PY) -m rdkit_qa_v3.dataset --name full

train-v3:  # SFT warm-start; pass NAME= (dataset dir under data/v3/, default full)
	$(PY) -m rdkit_qa_v3.train --name $(or $(NAME),full)

train-v3-grpo:  # GRPO stage on top of the SFT adapter; pass SFT_ADAPTER=path NAME=
	$(PY_GRPO) -m rdkit_qa_v3.train_grpo --name $(or $(NAME),full) --sft-adapter $(SFT_ADAPTER)

chat-v3:  # interactive local chat, live PubChem name lookup; pass ADAPTER=path
	$(PY) rdkit_qa_v3/chat_v3.py --adapter $(ADAPTER)

eval-v3:  # v3 benchmark; pass NAME= ADAPTER= TAG=
	$(PY) -m rdkit_qa_v3.eval --name $(or $(NAME),dev) --adapter $(ADAPTER) --tag $(TAG)

clean:  # remove generated data + results (keeps cached raw_bbbp.csv, data/v3/name_to_smiles.json)
	rm -rf data/dev data/full data/v2 data/v3/dev data/v3/full results artifacts/models
