# Code map

Use `scripts/reproduce.py` for normal reproduction; the modules below are the
smaller scientific building blocks behind that interface.

## Mechanical core

- `mechanical/netgen.py`: periodic random-Delaunay construction.
- `mechanical/topology.py`: spatial and nonspatial network factories.
- `mechanical/learners.py`: response solves and adjoint gradients.
- `mechanical/learners_local.py`: local material-update loop.
- `mechanical/objectives.py`: spectral response objectives.
- `mechanical/regularizers.py` and `constraints.py`: optional update terms.

The core contains no artifact rendering or release logic.

## Retained experiments

- `publication/scalar_gap.py`: scalar exemplar and gradient check.
- `publication/scalar_validation.py`: scalar ensembles and controls.
- `publication/bloch_gap.py`: periodic-vector training and Bloch validation.
- `publication/propagation.py`: post-training source-response diagnostic.
- `publication/nonspatial.py`: Watts--Strogatz topology screen.
- `publication/render_figures.py`: primary analysis-figure assembly.
- `publication/style.py`: shared figure styling.
- `publication/paths.py`: repository, data, and figure paths.

The production scalar and periodic-vector experiments use the same
response-conditioned log-stiffness update. Its local drive is analytically
bounded by one; the retained runs therefore use neither an inflation term nor
gradient clipping. Older radius-Euler and regularizer comparisons remain
available only as explicitly selected diagnostic modes.

`publication/two_time.py` and `publication/render_tables.py` preserve inactive
analyses from earlier drafts.  They are not called by `scripts/reproduce.py`
and their outputs are not included in the curated release archive.

Every module has a command-line interface. For example:

```sh
PYTHONPATH=scripts python -m publication.bloch_gap --help
```

## Validation and diagnostics

`validation/` contains required code/data release checks and the curated-data
archive builder. `diagnostics/` contains optional investigations that are not
needed by the main workflow.

`experiments/bloch_gap_3d.py` is an isolated dimension-extension test.  It uses
three displacement components, three-dimensional periodic Delaunay cells, and
a full cubic Brillouin-zone check. A single refined test is run with

```sh
PYTHONPATH=scripts python -m experiments.bloch_gap_3d --mode single
```

and a parallel seed ensemble with

```sh
PYTHONPATH=scripts python -m experiments.bloch_gap_3d --mode ensemble \
  --seeds 0 1 2 3 4 5 6 7 --workers 8
```

A saved candidate can be checked without retraining on successively denser
Brillouin-zone grids, followed by continuous band-edge refinement, with

```sh
PYTHONPATH=scripts python -m experiments.bloch_gap_3d --mode verify \
  --input scripts/outputs/experimental_3d/bloch3d_s2_net1_train1001.npz
```

Generated numerical caches live under the ignored `scripts/outputs/` directory.
Set `LSL_DATA_DIR` before importing a publication module to use another location.
