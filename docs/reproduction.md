# Reproducing the numerical results

## Reproducibility levels

The repository separates fast verification from expensive fixed-seed training:

1. `make check` restores and verifies the curated archive, then checks the
   deterministic fixtures, reciprocal gradients, and retained numerical results.
2. `make artifacts` deterministically renders plots and table fragments from the
   archived data into the ignored `artifacts/` directory.
3. `make reproduce WORKERS=N` reruns the retained experiments and then renders
   the derived artifacts. This complete path can take several hours.

No manuscript source or rendered paper is required by any workflow.

## Environment

The reference environment is Python 3.8 with the exact packages in
`requirements-lock.txt`. Later compatible versions may work. Matplotlib uses a
noninteractive backend and BLAS thread counts default to one so parallel jobs do
not oversubscribe the machine. The Python workflows use SciPy SuperLU by default
and require no proprietary solver.

## Data archive

`data/publication-data.tar.gz` contains the numerical inputs used by validation
and rendering. `data/manifest.json` records the path, byte size, and SHA-256
digest of every member and the archive itself. The archive preserves paths below
`scripts/outputs/` so experiment provenance and renderer inputs stay aligned.

It contains:

- the scalar exemplar, analytic-gradient check, ensembles, and controls;
- periodic-vector training, Brillouin-zone refinement at side lengths 5 and
  20, size ensembles through side length 20, and signal/material ablations;
- the nonspatial Watts--Strogatz screen;
- propagation spectra, convergence tests, source ensembles, and trained cells;
- scalar and periodic damping scans, shared absolute-window runs, the A--B--A
  retargeting trajectory, and the stiffness-calibration-disorder screen.

It excludes exploratory parameter searches, superseded experiments, diagnostic
images, console logs, and reduced smoke-test caches.

Verify, extract, or rebuild it with:

```sh
PYTHONPATH=scripts python -m validation.data_archive check
PYTHONPATH=scripts python -m validation.data_archive extract
PYTHONPATH=scripts python -m validation.data_archive build
```

## Commands and outputs

`scripts/reproduce.py` is the orchestration entry point; the root `Makefile`
provides short aliases.

| Command | Purpose | Principal output |
| --- | --- | --- |
| `make check` | Verify archive integrity and retained values | PASS/FAIL report |
| `make artifacts` | Render from archived data | ignored `artifacts/figures/` |
| `make experiments` | Rerun retained numerical evidence | ignored `scripts/outputs/` |
| `make reproduce` | Full experiments-to-artifacts workflow | data caches and artifacts |
| `make smoke` | Reduced propagation end-to-end test | ignored smoke-test caches |

Fixed seeds and scientific parameters are visible in `scripts/reproduce.py`.
The scalar and periodic-vector experiments share the bounded,
response-conditioned log-stiffness law with `eta = 0.02` and
`Lambda = 0.025 * (omega_hi**2 - omega_lo**2)`.

## Damped and scheduled learning

For mass-proportional damping, pass `damping="viscous"` and an absolute
`damping_gamma` to `mechanical.learners_local.train_local`. The same reciprocal
network is re-driven by the phase-conjugated response; a Bloch probe also
reverses `k` to `-k`. Run the deterministic adjoint, finite-difference, trace,
and zero-damping checks with:

```sh
PYTHONPATH=scripts python -m validation.damped_reciprocity
```

The scalar learner accepts `target_window=(lo, hi)` or an abrupt
`target_window_schedule=[(start_step, lo, hi), ...]`.
`spectrum_diagnostics=False` prevents eigensolves inside the learner; stored
material snapshots can be analyzed after training.

## Expected central checks

The fast check verifies, among other invariants:

- the scalar exemplar clears its prescribed spectral window;
- the analytic scalar gradient has maximum relative error below `2e-6`;
- the refined periodic-vector gap lies between bands 25 and 26 and has relative
  width approximately 0.381;
- every retained sample at cell sizes 5, 6, 8, 10, 12, 16, and 20 succeeds;
- the separately saved side-length-20 exemplar retains a positive complete gap
  under adaptive band-edge refinement;
- the propagation dataset contains ten materials and 500 coordinate sources;
- all retained scalar and periodic damping runs succeed over the tested range;
- all 30 shared absolute-window runs use zero online eigensolves and succeed;
- the scheduled run clears the active interval after each phase;
- nine of ten unknown-calibration runs clear their prescribed windows.

## External data location

Set `LSL_DATA_DIR` before running a workflow. An external directory must already
contain the same family subdirectories as `scripts/outputs/`; automatic archive
extraction is used only for the default location.
