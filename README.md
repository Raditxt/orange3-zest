# Orange3-Zest

[![tests](https://github.com/Raditxt/orange3-zest/actions/workflows/tests.yml/badge.svg)](https://github.com/Raditxt/orange3-zest/actions/workflows/tests.yml)

An [Orange Data Mining](https://orangedatamining.com/) add-on that brings research
models, especially ones Orange does not have yet (such as metaheuristic algorithms), into
widgets so they can be used without writing code.

Internship project, developed together with an AI assistant (Claude Code). Working
instructions for the AI are in [`CLAUDE.md`](CLAUDE.md).

## Status

| Stage | Scope | Status |
|---|---|---|
| 1 | Proof of concept: a pure-NumPy PCA whose results are identical to Orange's built-in PCA | Done |
| 2 | First model from the reference list (candidates: Firefly, Komodo Mlipir, Al-Biruni Earth Radius, PSO, extended decision tree), as a widget | Reviewing candidates |

Stage 1 reimplements PCA, which Orange already has. The goal is to show that an
algorithm can be rewritten outside Orange and its results proven to be the same, before
moving on to models Orange does not have. The result is wrapped as the **Zest PCA**
widget, which also serves as the template for the next model widgets.

## Installing the add-on

The add-on is not on PyPI yet. Install it from the repo folder into **Orange's bundled
Python** (replace `D:\Orange` with your Orange installation folder):

```powershell
D:\Orange\python.exe -m pip install -e .
```

Close and reopen Orange. The widget panel shows a **Zest** category containing **Zest
PCA** (screenshot: [`evidence/pca/zest_category.png`](evidence/pca/zest_category.png)).

## Zest PCA widget

Wraps `zestcore.pca` (pure NumPy) with the same inputs and outputs as Orange's PCA
widget:

- Input `Data`.
- Output `Transformed Data`: scores PC1..PCk, carrying over the original class and metas.
  Each PC carries its variance ratio. The output domain can be used by Apply Domain for
  new data.
- Output `Components`: one row per component, one column per feature, plus the metas
  `components` and `variance`.
- Settings: Components (1-100, default 2) and Normalize variables (default ON).
  Preprocessing (imputation, Continuize, Normalize) is done by Orange preprocessors, the
  same as in the PCA widget ([ADR 0002](docs/adr/0002-batas-preprocessing.md),
  [ADR 0003](docs/adr/0003-output-widget-dan-transform.md)).

**Claim:** Zest PCA results are identical to Orange 3.40.0 PCA **within the spec
tolerance** (`np.allclose`, rtol=1e-5, atol=1e-8; rules for tied components and the null
space in the [specification](docs/orange-pca-internals.md#test-rules)) **for X processed
by Orange preprocessors**. The claim is tested in `tests/test_owzestpca.py` (widget vs
widget) and demonstrated in the GUI on the heart_disease case. Maximum score difference
8.88e-16, components 25 × 25 and scores 303 × 25, see
[`evidence/pca/`](evidence/pca/README.md#gui-evidence-for-t2a-zest-pca-vs-orange-pca-heart_disease-27-nov-2026).

**Data shape limit ([ADR 0001](docs/adr/0001-jalur-dekomposisi-pca.md)):** zestcore only
reproduces the `covariance_eigh` path, which Orange uses when p ≤ 1000 and n ≥ 10p (n
rows, p columns after preprocessing). Outside that range Orange uses `full` or
`randomized` SVD, so Zest PCA results are only approximate and not guaranteed to be
identical. The widget shows a yellow warning with that message, but still sends its
outputs.

**Known limitations:**

- No scree plot (explained variance chart) yet. Planned as a follow-up task.
- No "explained variance" spin, no `Data` and `PCA` (projector) outputs, and no
  auto-apply button as in Orange's PCA widget.
- A failed fit (for example fewer than 2 rows, or no attributes left after preprocessing)
  shows as a red error with the raw message from NumPy/scikit-learn.
- The fit runs synchronously (no thread), which is fast enough for data the size of
  bank-full (45211 × 51).

## Stage 1 results

`zestcore.pca.pca` compared with Orange 3.40.0 PCA (PCA widget, Normalize ON) on three
datasets. The numbers below are the maximum absolute difference from Orange:

| Dataset | Compared | components | scores | `explained_variance_` |
|---|---|---|---|---|
| iris (150×4) | zestcore | 0.0 | 5.55e-17 | 0.0 |
| heart_disease (303×25) | zestcore | 0.0 | 8.88e-16 | 0.0 |
| bank-full (45211×51) | zestcore | 0.0 | 3.11e-15 | 0.0 |

Components and variances match exactly (0.0). Scores differ on the order of 1e-15,
which is floating-point rounding error from the order of BLAS operations, far below the
test tolerance (`np.allclose`, rtol=1e-5, atol=1e-8). The other references (scikit-learn
directly and recorded Orange GUI output), the handling of component signs, and
components in the null space are covered in
[`docs/orange-pca-internals.md`](docs/orange-pca-internals.md) and
[`notebooks/02_pca_perbandingan.ipynb`](notebooks/02_pca_perbandingan.ipynb).

## Repo structure

Two layers, with a strictly enforced boundary:

| Folder | Contents |
|---|---|
| `zestcore/` | Algorithm core, **pure NumPy**. `import Orange` and `import sklearn` are not allowed, so the algorithm can be inspected and tested without Orange. |
| `orangecontrib/zest/` | Orange layer that only wraps `zestcore`: the `pca.py` adapter (Orange projector) and `widgets/` (Zest PCA). |
| `tests/` | pytest. Orange and scikit-learn are used here as references. |
| `notebooks/` | Evidence notebooks: `01_pca_gui_evidence` (GUI vs script), `02_pca_perbandingan` (zestcore, scikit-learn, GUI vs Orange). |
| `evidence/pca/` | Output of the PCA and Zest PCA widgets recorded from the Orange GUI (`.pkl`, `.npy`, CSV, `.ows`, screenshots). See its [README](evidence/pca/README.md). |
| `docs/` | Orange PCA specification and design decisions (ADRs). |

### Documentation

- [`docs/orange-pca-internals.md`](docs/orange-pca-internals.md): specification of how
  Orange 3.40.0 runs PCA, the steps reproduced, test rules, scope limits, a claims table
  with `file:line` references, and implementation status.
- [`docs/adr/0001-jalur-dekomposisi-pca.md`](docs/adr/0001-jalur-dekomposisi-pca.md):
  decomposition path (covariance_eigh vs full SVD).
- [`docs/adr/0002-batas-preprocessing.md`](docs/adr/0002-batas-preprocessing.md):
  preprocessing responsibility boundary (zestcore vs caller).
- [`docs/adr/0003-output-widget-dan-transform.md`](docs/adr/0003-output-widget-dan-transform.md):
  widget outputs and transforming new data (Orange projector adapter).
- [`docs/proofs/pca_internals_check.py`](docs/proofs/pca_internals_check.py): script that
  checks every claim in the specification.

## Usage example

`zestcore` takes a matrix already processed as in Orange (Continuize, imputation, and
Normalize if used; see ADR 0002):

```python
import numpy as np
from zestcore.pca import pca

X = np.random.default_rng(0).normal(size=(200, 5))
res = pca(X)                      # k = min(n, p)
res.components                    # (k, p), one row per component
res.explained_variance_ratio      # (k,)
res.scores                        # (n, k)
```

The sign of each component is made deterministic: the element with the largest absolute
value is always positive, the same as the `svd_flip` convention in scikit-learn.

## Running the tests

All numbers are produced with Orange's bundled Python so they match the GUI:

| | Version |
|---|---|
| Orange | 3.40.0 |
| Python | 3.12.12 |
| NumPy | 2.1.3 |
| scikit-learn | 1.5.2 |

From the repo root (Windows, PowerShell; replace `D:\Orange` with your Orange
installation folder):

```powershell
D:\Orange\python.exe -m pip install -e . --no-deps
D:\Orange\python.exe -m pip install pytest
D:\Orange\python.exe -m pytest tests -q
```

Or in a regular Python 3.12 environment, with the same versions as CI:

```bash
pip install numpy==2.1.3 scikit-learn==1.5.2 Orange3==3.40.0 PyQt5==5.15.11 PyQtWebEngine==5.15.7 pytest
pip install -e . --no-deps
QT_QPA_PLATFORM=offscreen pytest tests -q
```

### bank-full data

The [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
dataset is not stored in the repo. Tests that use it read `bank-full.csv` from the path
in the `ZEST_BANK_FULL` environment variable and are skipped automatically if the file is
missing. In CI these tests are always skipped (62 passed, 9 skipped); with the data, all
71 tests pass.

```powershell
$env:ZEST_BANK_FULL = "C:\path\to\bank-full.csv"
```

Lint (rules in `pyproject.toml`, `[tool.ruff.lint]`):

```powershell
ruff check .
```

How to re-run the notebooks is described in
[`evidence/pca/README.md`](evidence/pca/README.md#running-tests-and-notebooks).

## License

[GPL-3.0-or-later](LICENSE), the same as Orange.
