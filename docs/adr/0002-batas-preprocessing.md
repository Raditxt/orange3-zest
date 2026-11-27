---
status: accepted
---

# Preprocessing responsibility boundary (zestcore vs caller)

## Context

Before PCA, the Orange widget runs Continuize → SklImpute → Normalize
(`docs/orange-pca-internals.md`, steps 1-3, Claims 1-3 and 7). These steps change the
data substantially: heart_disease goes from 13 columns to 25, 6 NaNs are imputed, and
imputed indicator values become fractions (`thal`: `[0.5515, 0.3887, 0.0598]`).
`CLAUDE.md` rule: `zestcore/` is pure NumPy, `import Orange`/`sklearn` is forbidden;
`orangecontrib/zest/` only wraps `zestcore`.

Details that must match exactly if preprocessing is reproduced:

- Continuize needs variable metadata: column type and the list of *declared* values
  (order of indicator columns and the drop rule `len(var.values) < 2`,
  `continuize.py:21`). Declared values that do not occur in the data still become a
  column of 0 (Claim 7).
- Mean imputation, then columns that are entirely NaN are dropped.
- Normalize: SD with ddof=0, no centering, SD 0/NaN → 1. A wrong ddof is not caught by
  the `components_`/ratio tests, only by the score and `explained_variance_` tests
  (Claim 3).

This decision is also a precedent for the Stage 2 models (metaheuristics), which will
need similar preprocessing.

## Options

1. **zestcore does everything (Continuize, imputation, Normalize).** Input: raw X +
   column metadata as plain NumPy/Python. zestcore can be used outside Orange with
   equivalent results. Most code; all the rules above must be reproduced and tested.
2. **The caller does everything.** The widget layer uses Orange's own preprocessors;
   zestcore only takes ready-to-use numeric X (steps 4-7). Preprocessing agreement is
   guaranteed because the code is Orange's code. zestcore alone is not equivalent to
   Orange PCA unless its user processes the data the same way.
3. **Split: caller does Continuize, zestcore does imputation + Normalize.** The part that
   needs Orange variable metadata stays in Orange; the numeric steps are in zestcore.

## Consequences

- Option 1: the Stage 1 claim ("pure-NumPy PCA identical to Orange") holds from raw
  data. zestcore needs its own representation of column metadata.
- Option 2: smallest zestcore core; the "identical" claim holds from already processed
  X. Tests must feed X preprocessed by Orange into zestcore.
- Option 3: the boundary sits in the middle of the pipeline, so both layers must agree
  on column order and NaNs; tests need to cover the seam.
- The formula for the number of zero-variance directions (test rule 2) needs the
  declared-values metadata; under options 2 and 3 that formula is computed on the
  caller or test side.

## Decision

I chose option 2: the caller does all preprocessing.

The widget layer and the tests use Orange's own preprocessors (Continuize → SklImpute →
Normalize). zestcore only takes ready-to-use numeric X and does steps 4-7:
n_components, centering + covariance with ddof=1 + eigh, the `svd_flip` sign, and the
scores. This way preprocessing agreement is guaranteed by using Orange's code, and the
CLAUDE.md rule (zestcore is pure NumPy, no `import Orange`/`sklearn`) holds without
reproducing Orange variable metadata.

The Stage 1 claim is stated as: "The pure-NumPy PCA in zestcore is identical to Orange
PCA for X already processed by Orange preprocessors." Tests feed X preprocessed by
Orange into zestcore, and the formula for the number of zero-variance directions is
computed on the test side. For Stage 2 models this is the precedent: zestcore takes
ready-to-use numeric data, unless a model genuinely needs special preprocessing.
