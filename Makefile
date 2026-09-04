PYTHON ?= python
WORKERS ?= 40
REPRODUCE = $(PYTHON) scripts/reproduce.py

.NOTPARALLEL:

.PHONY: all artifacts figures experiments reproduce check lint smoke

# The default target verifies the code and curated data.
all: check

artifacts:
	$(REPRODUCE) artifacts

figures: artifacts

# Recompute all numerical evidence but do not render or compile.
experiments:
	$(REPRODUCE) experiments --workers $(WORKERS)

# Full experiments-to-artifacts workflow; potentially several hours.
reproduce:
	$(REPRODUCE) full --workers $(WORKERS)

check:
	$(REPRODUCE) check

lint:
	ruff check scripts

smoke:
	PYTHONPATH=scripts $(PYTHON) -m publication.propagation smoke --workers $(WORKERS)
