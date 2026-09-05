# Local Spectral Learning

This repository contains the software and curated numerical data for training
reciprocal spring networks with self-probed trace learning. The paired-response
protocol supplies an edge-local gradient signal that can open an interior
spectral gap without selecting a source, receiver, or transmission path.

Manuscript source, submission documents, and rendered papers are intentionally
not stored in this repository or its Git history.

## Quick start

Python 3.8 or newer is supported. From a clean checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
make check
```

`make check` restores `data/publication-data.tar.gz` when needed, verifies its
manifest, imports the active modules, checks deterministic network fixtures and
reciprocal gradients, and validates the retained numerical results.

To regenerate plots and table fragments from the archived data, run:

```sh
make artifacts
```

Generated files are written beneath the ignored `artifacts/` directory. To
recompute all numerical evidence from fixed seeds before rendering artifacts:

```sh
make reproduce WORKERS=40
```

The full workflow can take several hours. See
[`docs/reproduction.md`](docs/reproduction.md) for parameters, data provenance,
and troubleshooting details. The Make targets delegate to the single
orchestration entry point, `scripts/reproduce.py`.

## Repository map

```text
scripts/mechanical/     network, response, learning, and objective code
scripts/publication/    fixed-seed experiments and artifact renderers
scripts/validation/     data-integrity and numerical-result checks
scripts/diagnostics/    optional analyses outside the main workflow
scripts/experiments/    isolated dimension-extension experiments
data/                   curated checksummed numerical archive
docs/                   code and data reproduction documentation
```

The working cache at `scripts/outputs/` and rendered files under `artifacts/`
are ignored. The curated archive contains only the 116 numerical files used by
the active validation and rendering workflows; exploratory searches, logs,
smoke-test caches, and superseded outputs are excluded.

## Commands

```sh
make check        # verify code, archive integrity, and retained results
make artifacts    # archived data -> ignored figures and table fragments
make experiments  # recompute retained numerical evidence
make reproduce    # experiments, then artifact rendering
make smoke        # reduced propagation calculation
make lint         # static checks
```

Set `LSL_DATA_DIR=/path/to/data` to use an external working-data directory and
`WORKERS=N` to control parallel calculations.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Source code is released
under the MIT License; the curated numerical data are released under CC BY 4.0.
See [`LICENSES.md`](LICENSES.md) for the exact scope.
