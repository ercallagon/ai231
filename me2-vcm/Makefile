# Makefile — convenience targets for the Hey Swift pipeline.
# The canonical driver is ./run.sh; these wrap it and add dev targets.

PYTHON ?= python
TAG    ?= hf_aligned_v1_a10_scratch
CONFIG ?= configs/vcm_hf_aligned_v1_a10.yaml
export PYTHONPATH := src

.PHONY: help install install-all download manifests features validate-data train export parity test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:            ## Install runtime deps (pip)
	$(PYTHON) -m pip install -r requirements.txt

install-uv:         ## Install runtime deps (uv, preferred)
	uv sync

install-all:        ## Install all extras (train, audio, pi, dev)
	$(PYTHON) -m pip install -e ".[all]"

download:           ## Fetch the SHA-pinned HF dataset (core splits)
	$(PYTHON) scripts/01_download_hf.py

download-full:      ## Fetch HF dataset incl. supplemental + synth negatives
	$(PYTHON) scripts/01_download_hf.py

manifests:          ## Build corrected versioned manifests (refuses overwrite)
	$(PYTHON) scripts/prepare_vcm_protocol.py --name hf_aligned_v1

features:           ## Build corrected source-aware clean mel cache
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) scripts/prepare_vcm_protocol.py --name hf_aligned_v1 --cache --workers 4

validate-data:      ## Validate corrected manifests and feature cache
	$(PYTHON) scripts/validate_vcm_protocol.py --config $(CONFIG)

train: validate-data ## Train corrected VCM (select CUDA_VISIBLE_DEVICES first)
	$(PYTHON) -m hey_swift.cli train vcm --tag $(TAG) --config $(CONFIG) --seed 42

export:             ## Export the trained VCM to ONNX
	$(PYTHON) scripts/03_export_onnx.py --tag $(TAG) --config $(CONFIG)

parity:             ## Verify torch <-> ONNX parity
	$(PYTHON) scripts/04_parity_check.py --tag $(TAG) --config $(CONFIG)

test:               ## Run the test suite
	$(PYTHON) -m pytest

lint:               ## Lint + type-check
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src || true

clean:              ## Remove caches (NOT data or experiments)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
