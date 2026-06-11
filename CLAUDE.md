# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a virtual environment at `~/python/venvs/exozippy_reu`. Activate it before running anything:

```bash
source ~/python/venvs/exozippy_reu/bin/activate
```

Install the package in editable mode if needed:

```bash
pip install -e source/
```

## Tests

Run all tests (slow tests included):
```bash
python -m pytest source/mmexofast/unit_tests/
```

Run fast tests only (skips grid searches and other slow integration tests):
```bash
python -m pytest source/mmexofast/unit_tests/ --fast
```

Run a single test file:
```bash
python -m pytest source/mmexofast/unit_tests/test_fit_types.py
```

Run a single test:
```bash
python -m pytest source/mmexofast/unit_tests/test_fit_types.py::TestClassName::test_method
```

The `--plot-grids` flag enables visual display of grid search plots during tests.

## Architecture

MMEXOFAST is a Python library for automated fitting of microlensing light curves, wrapping [MulensModel](https://github.com/rpoleski/MulensModel) and `sfit_minimizer`. It automates a staged fitting workflow from event detection through binary-lens modeling.

### Entry point

The public API is `MMEXOFASTFitter` (or the convenience wrapper `mmexofast.fit()`). The fitter accepts either pre-loaded `MulensModel.MulensData` objects or file paths, and a `fit_type` of `'point_lens'` or `'binary_lens'`.

```python
import mmexofast as mmexo
fitter = mmexo.MMEXOFASTFitter(files=['data.dat'], fit_type='point_lens')
results = fitter.fit()
```

### Workflow stages

`MMEXOFASTFitter.fit()` builds and executes an ordered list of `WorkflowStep` objects. Steps are organized into named **stages**:

| Stage | Key steps |
|---|---|
| `event_search` | `run_ef_grid` — EventFinder grid (Kim+2018) to locate t_0 |
| `fit_static_point_lens` | `est_pl_params`, `fit_pspl`, (optionally `fit_fspl`) |
| `fit_point_lens_parallax` | `fit_parallax_u0+`, `fit_parallax_u0-` |
| `renormalize` | `renormalize_datasets`, `refit_all` |
| `search_for_anomaly` | `compute_residuals`, `run_af_grid`, `get_anomaly_lc_params`, `classify_anomaly` |
| `fit_binary_lens` | `est_binary_params`, `fit_binary_models` (uses emcee) |
| `check_binary_renorm` | dynamically inserted if renormalization is needed post-binary-fit |
| `parallax_grids` | `run_parallax_grids` — full piE grid search |

Steps can be inserted dynamically (e.g., post-binary renormalization). The workflow can be interrupted with `stop_before`/`stop_after` and checkpointed/resumed via `restart_file`.

### Model identification: FitKey and labels

Every fit result is stored under a `FitKey` (frozen dataclass in `fit_types.py`). A `FitKey` encodes:
- `LensType` (POINT / BINARY)
- `SourceType` (POINT / FINITE)
- `ParallaxBranch` (NONE / U0_PLUS / U0_MINUS / U0_PP / U0_MM / U0_PM / U0_MP)
- `LensOrbMotion` (NONE / ORB_2D / KEPLER)
- `binary_model_type` (Wide, Close, CloseUpper, CloseLower, plus `_alt` variants)

Human-readable labels like `"PSPL static"`, `"FSPL par u0+"`, `"2L1S Wide static"` round-trip through `label_to_model_key()` / `model_key_to_label()` in `fit_types.py`.

### Results storage

`AllFitResults` (a `MutableMapping[FitKey, FitRecord]` in `results.py`) accumulates all fits. Each `FitRecord` holds best-fit params, sigmas, and a `full_result` wrapping either `MMEXOFASTFitResults` (sfit-based) or `EmceeFitResults` (emcee-based). Both implement the `BaseFitResults` ABC.

### Grid searches (`gridsearches.py`)

- `EventFinderGridSearch` — scans (t_0, t_eff) space to locate microlensing events.
- `AnomalyFinderGridSearch` — scans residuals from the best point-lens model to locate anomalies.
- `ParallaxGridSearch` — scans (piE_E, piE_N) space; supports coarse + fine refinement.

### Parameter estimation (`estimate_params.py`)

After the anomaly grid search, `AnomalyClassifier` (`classifier.py`) classifies the anomaly as `'close'`, `'wide'`, or `'high_mag'`. Based on the classification, one or more estimator classes (`WidePlanetGridSearchEstimator`, `ClosePlanetGridSearchEstimator`, etc.) seed initial binary-lens parameters for emcee.

### Model/Event construction (`mulens_object_config.py`)

`ModelConfig` and `EventConfig` are dataclasses that centralize all arguments to `MulensModel.Model` and `MulensModel.Event` construction. These are the single source of truth within a fitter instance. `EventConfig` is rebuilt after renormalization because renormalization replaces dataset objects.

### File naming convention

Data files follow `nYYYYMMDD.BAND.TELESCOPE.anything`. The `observatories.py` module parses filenames to set bandpass, ephemerides, and plot properties automatically.

### Output

`OutputConfig` controls optional file output (plots as PDFs, grid result text files, LaTeX/ASCII results tables, EXOZIPPy init JSON). Pass it as `output_config=` to `MMEXOFASTFitter`.

### Data

Sample data lives in `data/` (OB05390, OB08092, OB140939, OB161045, 2018DataChallenge). Unit test fixtures use `data/unit_test_data/`.
