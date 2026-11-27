---
status: accepted
---

# PCA core decomposition path (covariance_eigh vs full SVD)

## Context

Orange 3.40.0 PCA calls sklearn `PCA(svd_solver='auto')`. The solver chosen depends on
the shape of X after preprocessing (`sklearn/decomposition/_pca.py:531-543`;
specification: `docs/orange-pca-internals.md`, Claims 4 and 6):

- p ≤ 1000 and n ≥ 10p: `covariance_eigh` (eigendecomposition of the p × p covariance
  matrix). iris 150×4 and heart_disease 303×25 fall here.
- Otherwise, if min(n, p) ≤ 125: `full` (SVD of the centered X). Example: synthetic
  1000×125.
- min(n, p) ≥ 126, n < 10p, max(n, p) > 500: `randomized`, whose result depends on
  `random_state=0`. Examples: synthetic 1000×126, 1000×200, 600×130.

Relevant numbers:

- heart_disease, `covariance_eigh` vs `full`: components 1-18 `allclose` True
  (max 2.98e-13), `explained_variance_` 1-18 also True. Components 19-25 (null space, 7
  eigenvalues ~0) False (max 0.858), but those are not tested (test rule 2).
- `covariance_eigh` clips negative eigenvalues to 0 (`_pca.py:637`); heart_disease has 3
  eigenvalues clipped this way.
- `random_state` 0 vs 12345 does not change the result on either `covariance_eigh` or
  `full` (`array_equal` True).
- On both paths the sign is set by the same row-based `svd_flip` (`_pca.py:647`).

## Options

1. **`covariance_eigh` only.** One path, the same as what `auto` picks for iris and
   heart_disease. The covariance squares the condition number, so small eigenvalues are
   less accurate; for data with n < 10p the result is compared against Orange's `full`,
   and agreement has only been shown on one dataset (heart_disease).
2. **`full` SVD only.** One path, numerically more stable. For iris and heart_disease
   Orange itself uses `covariance_eigh`, so agreement depends on the difference between
   paths (2.98e-13 on heart_disease, far below the tolerance).
3. **Reproduce `auto` (two paths + selection rule).** Closest to Orange for every data
   shape. Twice the code and tests; sklearn's selection rule must be tracked whenever the
   sklearn version in Orange changes.

`randomized` coverage (applies to any option):

- **a. Limit zestcore to min(n, p) ≤ 125**, reject or warn outside that range. Simple,
  enough for iris and heart_disease; bank marketing needs to be checked first.
- **b. Reproduce `randomized` + `random_state=0`.** Would have to reproduce the random
  number sequence and iterations of sklearn's `randomized_svd` to be identical; far more
  complex.

## Consequences

- Option 1 or 2: data that Orange sends down another path is only *approximately*
  identical; the size of the difference must be measured per dataset, not assumed.
- Option 3: identical on the same path, but zestcore becomes tied to sklearn 1.5.2
  internal rules.
- Coverage a: zestcore cannot be used for wide data (min(n, p) ≥ 126) with n < 10p in
  Stage 1.
- Components in the null space are not identical under any option; tests restrict
  themselves to components with non-tied eigenvalues.

## Decision

I chose option 1 (`covariance_eigh` only) with coverage a.

The reason is that both Stage 1 datasets (iris 150×4 and heart_disease 303×25) are in
fact processed through `covariance_eigh` in Orange. With the same single path, zestcore
can be identical to Orange for this data, while the code and tests stay small, in line
with the Stage 1 target. I accept the reduced accuracy of small eigenvalues on the
covariance path because only components with non-tied eigenvalues are tested.

Data shape guard: if X does not satisfy p ≤ 1000 and n ≥ 10p, zestcore still computes
but emits a `warnings.warn` stating that Orange uses a different path (`full` or
`randomized`) for this shape, so the result is only approximate and not guaranteed to be
identical. `randomized` is not reproduced. Before use, the shape of a new dataset such
as bank marketing is checked after Continuize. If it falls outside the condition, the
difference is measured, not assumed. This decision is revisited if a Stage 2 model needs
wide data.
