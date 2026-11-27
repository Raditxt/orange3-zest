# Specification: how Orange 3.40.0 runs PCA

**Status: FINAL (20 Nov 2026).** Design decisions that depend on this specification:
`docs/adr/0001-jalur-dekomposisi-pca.md`, `docs/adr/0002-batas-preprocessing.md`.

Environment: `D:\Orange\python.exe` (Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2,
NumPy 2.1.3). Every `file:line` refers to the code installed in
`D:\Orange\Lib\site-packages\`, not GitHub master. Proof script:
`docs/proofs/pca_internals_check.py` (one `check_klaimN()` function per claim).
Tolerance for all comparisons: `np.allclose` default (rtol=1e-5, atol=1e-8) + the
maximum difference reported.

## Steps reproduced by zestcore (default widget path)

Stage 1 claim: the pure-NumPy PCA in zestcore is identical to Orange PCA for X already
processed by Orange preprocessors. Steps 1-3 are done by the **caller** (the widget layer
and the tests) with Orange's own preprocessors; zestcore only takes ready-to-use numeric
X and does steps 4-7 (ADR 0002). Steps 1-3 are recorded here so the caller and the tests
know what happens before X reaches zestcore.

1. **Continuize** (Claim 2). A discrete variable with ≥ 2 declared values becomes one
   0/1 column per value, ordered as in `var.values`; a binary variable also becomes 2
   columns. Fewer than 2 declared values: the variable is dropped (`continuize.py:21`).
   Missing value: NaN in all indicator columns of that variable. Continuous columns are
   unchanged.
2. **Mean imputation** (Claims 2, 7). NaN is replaced with the column mean; columns that
   are entirely NaN are dropped (`preprocess.py:150-173`). Imputed indicators become
   fractions (heart_disease, `thal`: `[0.5515, 0.3887, 0.0598]`), still summing to 1.
3. **Normalize, only when the Normalize option is ON** (widget default ON; Claims 1, 3).
   Divide each column by its SD with **ddof=0**, without subtracting the mean. SD = 0 or
   NaN is replaced with 1 (`normalize.py:49-54`).
4. **Number of components + shape guard** (Claims 4, 6; ADR 0001). Widget: k = min(n, p,
   100), with n and p counted *after* steps 1-3. The number of components chosen in the
   widget only slices the output. zestcore truncates to min(n, p); the number 100 is
   passed by the caller (`n_components=100`), not fixed in zestcore. If X does not
   satisfy p ≤ 1000 and n ≥ 10p, zestcore still computes but emits a `warnings.warn`:
   Orange uses a different path (`full` or `randomized`) for this shape, so the result
   is only approximate and not guaranteed to be identical.
5. **Fit** (Claims 3, 4, 6). Covariance with **ddof=1**, computed in exactly sklearn's
   form and multiplication order: C = X.T @ X, C −= (n·mean)·meanᵀ, C /= n−1
   (`_pca.py:611-617`). Mathematically equivalent to Xc.T @ Xc/(n−1) but the
   floating-point result differs; the (n·a)·b order is also followed because n·(a·b)
   changes the last digits of C. With this form C is exactly equal to Orange's;
   eigendecomposition of the covariance (`covariance_eigh` path only, ADR 0001); sort
   the eigenvalues in descending order; negative eigenvalues are clipped to 0.
   `explained_variance_` = eigenvalue. `explained_variance_ratio_` = eigenvalue / the
   sum of *all* min(n, p) eigenvalues before slicing to k (`_pca.py:652-653`); if
   k < min(n, p), the ratios sum to < 1.
6. **Sign** (Claim 5). For each row of `components_`, find the element with the largest
   absolute value (the first index on a tie) and multiply the row by its sign (row-based
   `svd_flip`, `extmath.py:848-905`).
7. **Scores**: X @ `components_`.T − mean @ `components_`.T, the sklearn `transform`
   form (`decomposition/_base.py:149, 156`, whiten=False), called by Orange through
   `Orange/projection/base.py:115-116`. It has no claim of its own; it is proven by the
   score tests.

## Test rules

1. **Compare the PC scores and `explained_variance_`**, not only `components_`/ratio. A
   wrong ddof in Normalize produces a uniform scale of sqrt((n-1)/n): `components_` and
   the ratio stay the same, while `explained_variance_` shifts by (n-1)/n = 0.99333 and
   the scores shift by at most 0.0110 (iris, Claim 3).
2. **Components in the null space are not compared.** The ~0 eigenvalues are tied, so
   their vectors are not unique: heart_disease components 1-18 `covariance_eigh` vs
   `full` True (max 2.98e-13), 19-25 False (max 0.858). The number of zero-variance
   directions (eigenvalue < 1e-10) is predicted on the test side from the data before
   the fit (needs Orange variable metadata, ADR 0002):

   zero count = Σ (declared values − values that occur + 1) over each discrete variable
   that is not dropped, + the number of constant continuous columns.

   If all declared values occur, the discrete term is 1 per variable. Dropped variables
   (fewer than 2 declared values, or entirely NaN) contribute 0. This formula is a lower
   bound: extra zero directions appear when n − 1 < p or when columns are collinear
   (including two indicator columns from different variables that always occur
   together). Confirmed on heart_disease (7 = 7) and synthetic data (6 = 6; the formula
   without the "values that do not occur" term predicts 4). Bank marketing (UCI):
   `bank-full` 9 = 9; `bank-additional-full` 11 vs a prediction of 10, the extra one
   coming from `housing=unknown` and `loan=unknown`, which occur in exactly the same 990
   rows.

   Null-space scores are still compared, with rtol=0 and atol=1e-8 (their values are
   ~1e-15 noise, so relative error is meaningless). The same rule is used for the
   null-space variance ratio (~1e-16) and the null-space `explained_variance_` (~1e-15,
   some clipped to 0).
3. **Sign of tied components** (approved by Raditya, 20 Nov 2026). Reference (Orange)
   row k is called tied if |largest| − |second largest| < 1e-10.
   - Not tied: `components_[k]` and the k-th score column are compared directly.
   - Tied: s = sign(a_k · b_k) from the components (a = reference, b = zestcore), then
     compare `a_k` with `s·b_k` **and** the k-th reference score with `s·` the k-th
     zestcore score (not tested separately with "or").
   - The tolerance stays the `np.allclose` default.
   - Tied rows also appear in the heart_disease null space (PC19, 22, 24, 25); not
     relevant because null-space components are not compared (rule 2).

## Known scope limits

- **Tied components from binary variables.** Continuize turns a binary variable into
  `x` and `1-x`; after normalization both loadings have the same magnitude with
  opposite signs, so `argmax` is decided by ~1e-16 error. heart_disease: PC2 (`gender`)
  and PC3 (`fasting blood sugar > 120`); iris: 0 of 4; bank-full: PC2, 4, 6, 7 (outside
  the null space, measured 23 Nov 2026). Shuffling the row order 20×: the sign of PC2
  flips 10×, PC3 2×, within Orange itself.
- **The `randomized` path.** If min(n, p) ≤ 125, `auto` only chooses `covariance_eigh`
  or `full`, and `random_state` has no effect. If min(n, p) ≥ 126, k = 100 <
  0.8·min(n, p) and `auto` chooses `randomized` when n < 10p and max(n, p) > 500
  (`_pca.py:539`; synthetic 1000×126, 1000×200, 600×130). `randomized` is not
  reproduced; outside the `covariance_eigh` condition the difference is measured, not
  assumed (ADR 0001).
- **Bank marketing** (UCI, checked 20 Nov 2026, data not stored in the repo; target `y`
  not included, "unknown" is a category, not NaN): `bank-full` 45211×16 → 45211×51,
  `bank-additional-full` 41188×20 → 41188×63 after Continuize. Both have n ≥ 10p, `auto`
  chooses `covariance_eigh`, so they fall within the scope of ADR 0001.

## Claims table

| No | Claim | Source (file:line) | REPL evidence | Result |
|----|-------|---------------------|------------|-------|
| 1 | `PCA.preprocessors = [Continuize(), SklImpute()]`; the widget appends `Normalize(center=False)` at the end (widget default `normalize=True`) | `Orange/projection/base.py:268-269`; `Orange/projection/pca.py:27`; `Orange/widgets/unsupervised/owpca.py:50, 277-283` | `check_klaim1()`: `OWPCA._create_projector` called with a mock object | **True.** Default: `[Continuize, SklImpute]`; widget: `[Continuize, SklImpute, Normalize]`, `center=False`, `norm_type=NormalizeBySD` |
| 2 | Continuize default = Indicators, zero_based; each discrete value becomes a 0/1 column, a binary variable becomes 2 columns | `Orange/preprocess/preprocess.py:47-48`; `Orange/preprocess/continuize.py:21-25, 45-60`; `Orange/preprocess/transformation.py:119-128, 152` | `check_klaim2()`: mini table 4×4 (age, binary sex, 3-valued color, constant with 1 value, 1 NaN) | **True.** 4 columns become 6: `umur, jk=L, jk=P, warna=merah, warna=hijau, warna=biru`. Also: a discrete variable with fewer than 2 values is dropped; a missing discrete value gives NaN in all of its indicator columns |
| 3 | SD in Normalize uses ddof=0; PCA covariance uses ddof=1 | `Orange/preprocess/normalize.py:25, 48, 53-58`; `Orange/statistics/util.py:352, 485` (`bn.nanvar`, ddof default 0); `sklearn/decomposition/_pca.py:591, 611-617` | `check_klaim3()` on iris (150×4) | **True.** Normalize vs X/std(ddof=0): True, max 1.78e-15; vs ddof=1: **False**, max 0.0340. `explained_variance_` vs eig cov ddof=1: True, max 6.01e-15; vs ddof=0: **False**, max 0.0282 |
| 4 | `svd_solver='auto'` chooses `covariance_eigh` if p ≤ 1000 and n ≥ 10p | `sklearn/decomposition/_pca.py:531-543` (selection), `611-643` (path body; negative eigenvalues clipped to 0 at `637`); `Orange/projection/pca.py:33` | `check_klaim4()`: widget projector (Normalize ON), read `model.proj._fit_svd_solver` | **True.** iris 150×4 (10p=40): `covariance_eigh`. heart_disease 303×25 (10p=250): `covariance_eigh`, 7 components < 1e-10 (`3.9e-15, 1.4e-15, 8.0e-16, 5.7e-16, 0, 0, 0`; the last three clipped to 0, explained in Claim 7). bank-full 45211×51 (10p=510): `covariance_eigh` (`notebooks/02_pca_perbandingan.ipynb`, summary table) |
| 5 | Sign: `svd_flip(u_based_decision=False)`, the \|largest\| element of each `components_` row is positive | `sklearn/utils/extmath.py:848-905` (`svd_flip`; the `else` branch = based on rows of `v`, `argmax` takes the first index on a tie); `sklearn/decomposition/_pca.py:647` (called for the `full` and `covariance_eigh` paths), `760, 773` (arpack/randomized paths). Orange does not flip signs itself (grep `sign\|flip` in `pca.py`, `base.py`, `owpca.py`: nothing) | `check_klaim5()`: widget path on iris & heart_disease; `covariance_eigh` vs `full`; tied row detection; shuffle row order 20× | **True, with a caveat.** All rows positive (iris 4/4, heart_disease 25/25). heart_disease components 1-18: smallest eigenvalue gap 0.0127 (not tied); `eigh` vs `full`: True, max 2.98e-13. Rows 19-25: False, max 0.858. **But** heart_disease PC2 and PC3 are *tied* (gap 6.7e-16, 8.9e-16): the two indicator columns of a binary variable (`gender`, `fasting blood sugar > 120`) have loadings of equal magnitude with opposite signs. Shuffling row order 20×: the sign of PC2 flips 10×, PC3 2× |
| 6 | Widget `PCA(n_components=100, random_state=0)`, truncated to `min(X.shape)` | `Orange/widgets/unsupervised/owpca.py:27` (`MAX_COMPONENTS = 100`), `278-279` (projector; `ncomponents` only stored in `projector.component`), `176-186` (fit once, ratio from the model), `326-354` (output only sliced `[:self.ncomponents]`); `Orange/projection/pca.py:38-41` (truncation, X after preprocessing); `sklearn/decomposition/_pca.py:738, 754, 771` (`random_state` only used by arpack/randomized), `698-705` (sklearn truncation) | `check_klaim6()`: projector parameters; iris, heart_disease, iris 3 rows; sklearn without truncation; `ncomponents` 2 vs 5; `random_state` 0 vs 12345 | **True.** iris 150×4 → 4 components; heart_disease 303×25 → 25 (from the 25 columns after Continuize, not 13); iris 3 rows 3×4 → 3 (truncated by n). Without truncation sklearn raises `ValueError`. `sum(ratio)` = 1.000000000000000 in all three. `ncomponents` 2 vs 5: `components_` exactly equal (`array_equal`). `random_state` 0 vs 12345: exactly equal on `covariance_eigh` and `full`. **Limit:** the actual truncation is `min(n, p, 100)`; synthetic data 1000×125 → `full`, 100 components; 1000×126, 1000×200, 600×130 → **`randomized`**, 100 components (`_pca.py:539`) |
| 7 | heart_disease: 6 NaNs, 7 discrete variables; 13 → 25 columns; 7 directions with ~0 variance = 1 per discrete variable | `Orange/preprocess/preprocess.py:150-173` (SklImpute mean, columns entirely NaN dropped); `Orange/preprocess/continuize.py:21, 45-60`; `Orange/preprocess/transformation.py:131-152`; `Orange/preprocess/normalize.py:49-54` (SD 0/NaN → 1) | `check_klaim7()`: type & NaN per attribute; columns per stage; imputed indicators; direction w per indicator group; 50-row synthetic data | **True.** NaN: `major vessels colored` 4, `thal` 2 (6 rows). 7 discrete + 6 continuous; 13 → 25 columns = 6 + (2+4+2+3+2+3+3). `thal` indicators in rows 88, 267 = `[0.5515, 0.3887, 0.0598]` (column means), sum per group = 1 (max difference 0). Direction w (w_j = sd_j on the group's columns): onto components 1-18 max 4.4e-15, remainder in 19-25 max 4.3e-15, rank 7, variance 0. Synthetic (constant continuous, value that does not occur, discrete entirely NaN, discrete with 1 value): 6 zero directions, corrected formula 6, original formula 4 |

## Implementation status (handoff 23 Nov 2026, afternoon)

This section is for the next session after `/clear`: reading this + CLAUDE.md is
enough.

### T2a: Zest PCA widget (update 27 Nov 2026)

Scope and definition of done are in the T2a task brief (Notion). The design is in
`docs/adr/0003-output-widget-dan-transform.md` (option 1c).

Done:

- CLAUDE.md "Current stage" includes the Zest PCA widget in scope (`13b2eb2`).
- ADR 0003 option 1c accepted (`f444838`).

**T2a done (27 Nov 2026).** Code (steps 1-4b), review since `cc0feb3` (p = 0 test and
the `commit()` guard, ruff rules), heart_disease GUI evidence by Raditya, and README
(installation, claim, ADR 0001 limit, limitations). Commits `076645f`..`5c86b9d`, CI
`5c86b9d` green: 62 passed, 9 skipped (bank-full); locally 71 passed. Lint `ruff check .`
clean.

- Step 4b done (27 Nov 2026): the `OWZestPCA` widget, 6 `/tdd` cycles, one commit per
  cycle (`9391049`..`85f12c9`). Files: `orangecontrib/zest/widgets/owzestpca.py`,
  `widgets/__init__.py` (Zest category), `widgets/icons/{ZestPCA,category}.svg`,
  `ShapeWarning` in `zestcore/pca.py`, `package-data` in `pyproject.toml`, test
  `tests/test_owzestpca.py` (`WidgetTest`, compared against the OWPCA widget with the
  same settings, full `compare()`, spec tolerance). The design did not change.

  | cycle | behavior | result |
  |---|---|---|
  | 1 | iris → Transformed Data + Components vs OWPCA: names, class/metas, `variance`, Components metas | (2 PCs, Normalize ON): components 0.0, scores 0.0, ratio 0.0; (4 PCs, OFF): 0.0 / 2.78e-17 / 0.0 |
  | 2 | `None` input clears both outputs | – |
  | 3 | `heart_disease[:100]` (100x25, n < 10p) → `Warning.shape_outside_spec` containing the `ShapeWarning` text; outputs still sent; cleared when iris comes in | – |
  | 4 | `iris[:1]` → `Error.fit_failed` ("minimal 2 baris"), outputs cleared; cleared when iris comes in | – |
  | 5 | train on even iris rows (4 PCs, ON), odd rows transformed to the Transformed Data domain | new-data scores 5.55e-17 |
  | 6 | `orange.widgets` entry point → `WidgetDiscovery` → `icon_loader`: Zest category, widget and category `.svg` icons, icon ≠ Orange's `PCA.svg` | – |

  Suite: 69 passed, 0 skipped (local, bank-full present). CI `85f12c9` green: 60
  passed, 9 skipped (bank-full tests, the data is not on the runner). Lint: ruff 0.17.6
  in a separate venv, `ruff check --isolated --select E4,E7,E9,F zestcore orangecontrib
  tests` clean (the old ruff default set; the ruff 0.17 default also turns on the import
  order rule I001 and RUF012, which older tests already violate, so that is a style
  matter and not yet a repo rule).
  Notes:
  - Cycle 5 was green immediately, because compute_value is carried by the 4a adapter.
    Sensitivity test: with Transformed Data built as a plain table without
    compute_value (ADR 0003 option 2), only the cycle 5 test fails.
  - `package-data` was checked by building a wheel from a clean copy: with that section
    both SVGs are included, without it there is no SVG. Building in the repo folder was
    misleading at first, because `build/` and `SOURCES.txt` left over from an earlier
    build still carried the SVGs.
  - The `commit()` guard for when there is no fit result yet (spin changed without data)
    got a test after the 27 Nov 2026 review (`test_spin_diubah_sebelum_ada_data`). The
    test calls `commit()` directly, because an exception in a Qt callback makes the
    pytest process abort (0xC0000409) rather than fail the test. Sensitivity test:
    without the guard, the test fails with `AttributeError`.
  - `Error.fit_failed` still only catches `ValueError` (decision 2). Probe 27 Nov 2026:
    p = 0 (class only, or a discrete variable with 1 value), n = 0, columns that are all
    NaN, and sparse data all end in `ValueError`, so they are caught without a crash.
    NumPy's `LinAlgError` is also a subclass of `ValueError`. Test for p = 0:
    `test_tanpa_atribut_tampil_sebagai_error`. The message is still raw sklearn/NumPy
    text (for example "attempt to get argmax of an empty sequence" for a NaN column).
    Constant data does not error, but the variance ratio becomes `nan` (0/0 in
    `zestcore/pca.py`). Neither is handled yet.
  - Lint: `[tool.ruff.lint]` in `pyproject.toml` (`E4, E7, E9, F`, E402 excluded for
    notebooks), so `ruff check .` is enough.
  - `self.widget` is set in `setUp`, so the built-in `WidgetTest` checks (minimum size,
    image export, `Msg` base class) also run.
  - `zestcore/pca.py:41` (`if n < 2: raise ...` on one line, ruff E701) was split in
    cycle 3.

#### GUI evidence for T2a: heart_disease case (done by Raditya, 27 Nov 2026)

Case question: **can the 25 columns of heart_disease (13 attributes, becoming 25 columns
after Continuize) be summarized into 2 PCs that still separate sick/healthy (`diameter
narrowing`)?** The same workflow was built twice, once with Orange PCA and once with
Zest PCA. The files are in `evidence/pca/` (full list in its README).

| Widget | Setting (read from `.ows`) |
|---|---|
| File | `heart_disease.tab` (built into Orange), 303 rows |
| PCA / Zest PCA | Components 25, Normalize ON |
| Select Columns | features PC1, PC2; target `diameter narrowing` |
| Logistic Regression | default (L2) |
| Test and Score | Cross validation 10 fold, stratified; one for 2 PCs, one for the 25 original columns (directly from File) |
| Scatter Plot | PC1 × PC2, colored by `diameter narrowing` |
| Data Sampler → Zest PCA (1) | only in `heart_pca_zest.ows`: Fixed sample size 100, stratified, fixed seed |

Results (Logistic Regression, 10-fold CV, the same in the Orange and Zest branches):

| Features | AUC | CA |
|---|---|---|
| 2 PCs (PC1, PC2) | 0.900 | 0.818 |
| 25 original columns | 0.909 | 0.845 |

Answer: **yes, with a small loss.** The two PCs carry only 29.3% of the variance
(PC1 0.193, PC2 0.100). Even so, AUC drops by 0.009 and CA by 0.027 compared with the
25 original columns. In the Scatter Plot the two classes separate mainly along PC1. PC1
roughly describes the heart's response during the exercise test. The largest |w| on
PC1 are `thal=normal` +0.317, `exerc ind ang` ±0.311, `slope peak exc ST=upsloping`
+0.294, `chest pain=asymptomatic` −0.293, `max HR` +0.284, and `ST by exercise` −0.279.
PC2 is dominated by `gender` (±0.472). This is a summary of weights, not a causal claim.

Zest vs Orange:

- **Full precision (main evidence):** the scores in
  `heart_zestpca_norm-on_25pc_scores.pkl` (Save Data) and the components in
  `heart_zestpca_norm-on_25pc_components.npy` (Python Script) are compared with
  `heart_pca_norm-on_25pc_*` (Orange PCA GUI evidence of 20 Nov 2026, same settings) by
  `evidence/pca/compare_zest_gui.py` (`compare()`, rtol=1e-5 atol=1e-8):
  `[norm-on_25pc] IDENTIK`, components (25, 25), scores (303, 25), max difference
  8.88e-16. This result is the same on Raditya's machine and when re-run on 27 Nov 2026.
  Not CSV, because CSV is rounded with `%g`.
- **GUI display (Raditya's observation):** the Components table, Test and Score, and the
  scatter plot are the same in both branches to the digits shown.
- **Category and icon:** `zest_category.png`, the Zest category with Zest PCA in the
  left panel.
- **ADR 0001 warning:** `heart_zestpca_warning.png`. On 100 rows (100×25, n < 10p) Zest
  PCA (1) shows the warning icon, and the outputs are still sent.

How to install the add-on into Orange: see the repo README, section "Installing the
add-on".

- Step 1 done (25 Nov 2026): `transform` in `zestcore/pca.py`, Seam A
  (`tests/test_zestcore_transform.py`) green. Max difference vs `model(test).X`:
  iris 0.0, heart_disease 0.0 (both bit-identical), bank-full 3.55e-15
  (np.allclose rtol=1e-5 atol=1e-8 passes). Suite: 50 passed.
- Step 2 done (25 Nov 2026): Seam B, a single new row `test[:1]` (1, p), iris and
  heart_disease. Green immediately without code changes; max difference 0.0 on both.
  1-D input is not supported and not tested.
- Step 3 done (25 Nov 2026): `pca()` calls `transform`, so the score formula lives in
  one place. Observation: pca `scores` before vs after the refactor are bit-identical on
  iris, heart_disease, bank-full (`n_components=100`). Suite: 52 passed.
- Step 4a done (25 Nov 2026): the adapter `orangecontrib/zest/pca.py`
  (`ZestPCA(Projector)`, `ZestPCAFit`, `ZestPCAModel(DomainProjection)`), test
  `tests/test_zest_pca.py`, 8 tests. proj form = a separate small object `ZestPCAFit`
  (ADR 0003 Implementation notes, evidence `docs/proofs/proj_forms_check.py`).
  Reference: the PCA widget projector, full `compare()`. Max score difference
  (components, expl. var, and ratio 0.0 in every case):

  | seam | dataset | scores |
  |---|---|---|
  | 1 full data, Normalize ON | iris / heart_disease / bank-full | 5.55e-17 / 8.88e-16 / 1.78e-15 |
  | 2 Normalize OFF, `ZestPCA(n_components=100)` | iris / heart_disease | 2.78e-17 / 5.68e-14 |
  | 3 Apply Domain | iris even→odd / heart_disease `[:250]`→`[250:]` | 5.55e-17 / 8.88e-16 |
  | 4 pickle round trip (output domain + model) | iris | 0.0 (`array_equal`) |

  `zero_variance_count` on the training subset = measured (iris 0, heart 7, Normalize
  ON/OFF). Seams 2-4 were green immediately because the adapter was already complete in
  the first slice; sensitivity tests (temporary mutations in `pca.py`) make each of them
  fail. Findings:
  - `Table.transform` calls compute_value per chunk of 5000 rows
    (`data/table.py:385`): transform may only use statistics from the fit (template rule
    in ADR 0003).
  - iris scores 5.55e-17, not 0.0 as in Seam A: zestcore components are C-order,
    sklearn's are F-order, with exactly equal values. With an F-order copy the
    difference is 0.0. This matches the memory layout decision below, so the code was
    not changed.

  Suite: 60 passed. Next: 4b.
- Seam A: the widget projector is fitted on the even rows. The odd rows become new data
  with `model(test).X` as the reference. The zestcore input is
  `test.transform(model.pre_domain).X`, while `mean_`/`components_` are taken from
  `model.proj` on both sides. Assessment: `np.allclose` default on the whole score
  matrix, plus the maximum difference. This test deliberately does not use `compare()`
  or `zero_variance_count`.
- The reference side was checked before implementation:

  | dataset | n_train | p | Orange solver | new-data scores | NaN in new X |
  |---|---|---|---|---|---|
  | iris | 75 | 4 | covariance_eigh | (75, 4) | 0 |
  | heart_disease | 152 | 25 | full (152 < 10p = 250) | (151, 25) | 0 |
  | bank-full | 22606 | 51 | covariance_eigh | (22605, 51) | 0 |

- To keep in mind: the `model(test).X` reference goes through `Table.transform`. On
  bank-full this path once differed by ~1.8e-15 from `proj.transform` (see Open
  questions). A non-zero maximum difference on bank is expected, since it is far below
  the tolerance.

Next (step by step, wait for Raditya at each step):

1. Minimal implementation of `transform` in `zestcore/pca.py` with the formula
   `X @ C.T - mean @ C.T` → green. Report the maximum difference per dataset.
2. Seam B: input of shape (1, p) must work. 1-D input is not supported. `pca()` rejects
   n < 2, `transform` must not reject it too.
3. Refactor `pca()` to call `transform`, so the formula lives in one place. The
   `test_zestcore_pca.py` tests must stay green.
4. Step 4, split in two:
   - **4a. Adapter without widget** (done, see above): `ZestPCA(Projector)` +
     `ZestPCAModel(DomainProjection)` in `orangecontrib/zest/`. Start with a small
     example to decide the form of the `proj` object (ZestPCA itself like
     `LinearProjector`, or a separate small object). Then `/tdd`:
     - full data vs OWPCA with full `compare()`;
     - Apply Domain (new data): iris with an even/odd split, or heart_disease with
       n_train ≥ 250 (10p, p = 25), so Orange stays on the covariance_eigh path;
     - pickle round trip (output domain and model);
     - Normalize OFF.
   - **4b. Widget** (done 27 Nov 2026, see above): `OWZestPCA` + `WidgetTest`. The ADR
     0001 warning from `zestcore.pca` (shape outside p ≤ 1000 and n ≥ 10p) is shown as a
     widget `Msg`. Widget icon and category. Design approved 25 Nov 2026 (see below).

#### OWZestPCA design (4b, approved 25 Nov 2026)

Files:

- `orangecontrib/zest/widgets/owzestpca.py`: the widget.
- `orangecontrib/zest/widgets/icons/ZestPCA.svg` (widget icon, different from Orange's
  `PCA.svg`) and `icons/category.svg` (category icon).
- `orangecontrib/zest/widgets/__init__.py`: category metadata `NAME = "Zest"`,
  `DESCRIPTION`, `BACKGROUND`, `ICON`.
- `pyproject.toml`: `[tool.setuptools.package-data]` for `icons/*.svg`. An editable
  install works without it, a wheel/non-editable install does not.
- `tests/test_owzestpca.py`: `WidgetTest`.

Inputs/Outputs:

- Input `Data` (Table).
- Outputs `Transformed Data` and `Components` (`dynamic=False`), mirroring
  `owpca.py:316-355`. Transformed Data = `model(data)` sliced to the first
  `ncomponents` attributes, carrying the original class and metas, with each PC given
  `var.attributes["variance"] = round(ratio, 6)`. Components = rows
  `components_[:ncomponents]`, columns = the `orig_domain` attribute names with
  `compute_value=lambda _: None`, metas `components` (PC1..) and `variance`.

Settings: `ncomponents = Setting(2)` (spin 1-100, default the same as OWPCA) and
`normalize = Setting(True)`.

Out of scope (recorded in the README): scree plot, explained variance spin, `Data`
output, `PCA` (projector) output, auto-apply button.

Flow:

- Data or `normalize` changes → fit:
  `ZestPCA(n_components=100, preprocessors=ZestPCA.preprocessors + [Normalize(center=False)]
  if checked)` (`owpca.py:278-282`), then `model = projector(data)` and
  `transformed = model(data)` (the compute_value path, `owpca.py:321`). Both inside
  `warnings.catch_warnings(record=True)`.
- `ncomponents` changes → only re-slice the existing result, no refit.

Raditya's decisions (25 Nov 2026):

1. **`ShapeWarning(UserWarning)` in `zestcore/pca.py`**, used as the category of the
   ADR 0001 `warnings.warn`. The widget only catches `ShapeWarning` and passes its
   message on as is through `Warning.shape_outside_spec = Msg("{}")`. The alternative
   (filtering `UserWarning` by message text) was rejected because it breaks if the text
   changes.
2. **`Error.fit_failed = Msg("{}")`** catches `ValueError` (for example zestcore
   rejecting n < 2) and clears the outputs. It does not mirror OWPCA's
   `no_features`/`no_instances`.
3. **No "All" value** in the components spin. `ncomponents` > k gives k components
   through ordinary slicing.

The fit is **synchronous**, without `ConcurrentWidgetMixin` (approved). Reason:
`catch_warnings` changes global state so it is not thread-safe, and the code is simpler;
the bank-full fit is only a 51×51 eigh. **This part is not a template to copy:** a
learner widget (ExtDT) may still need `ConcurrentWidgetMixin` if its fit is slow.

`/tdd` seams requested by Raditya (proposed in detail when the session starts): Zest
widget output vs OWPCA for the same data (Transformed Data, Components, variance),
slicing of the number of components, Normalize OFF, the warning `Msg` on
`heart_disease[:100]`, a `None` input clearing the outputs. The GUI evidence
(`.ows`/screenshots) is done by Raditya.

Suggested skills: `tdd` for steps 1-3. For step 4, `codebase-design` if the form of the
`proj` object needs the seam/adapter vocabulary, and `domain-modeling` if a new decision
deserves an ADR.

### Done

- **Pure-NumPy PCA core** (`1503b04`). `zestcore/pca.py`: `pca(X, n_components=None)` →
  `PCAResult(mean, components, explained_variance, explained_variance_ratio, scores)`.
  Steps 4-7, `covariance_eigh` path only (ADR 0001); X must already be processed by
  Orange preprocessors (ADR 0002). The widget uses `n_components=100`, passed by the
  caller. No `transform` for new data yet (added when the widget is built). Warning
  outside p ≤ 1000 and n ≥ 10p; `ValueError` if n < 2.
- **Covariance in exactly sklearn's order** (`d8f4fe9`). `n * np.outer(mean, mean)`
  replaced with `n * mean[:, None] * mean[None, :]` = (n·a)·b. As a result zestcore's
  components, `explained_variance_`, and ratio are exactly equal to Orange's (0.0) on
  all three datasets.
- **bank-full GUI evidence** (`b27afbd`). `evidence/pca/bank_pca_norm-on_51pc_*`
  (`.npy`, `.pkl`, Components CSV; no score CSV) + `.ows` + screenshots. The `.ows` was
  saved again: both Save Data with annotations OFF, `stored_name` matching the files in
  the repo.
- **Comparison notebook** (`4be7154`). `notebooks/02_pca_perbandingan.ipynb`: zestcore,
  scikit-learn, and the GUI vs Orange PCA (widget projector, Normalize ON) for iris,
  heart_disease, bank-full; max difference tables, tied components, zero directions, a
  memory layout evidence cell, the bank-full conclusion, the chapter IV narrative. The
  notebook is built with the `tests/` helpers (`compare`, `compare_variance`,
  `compare_ratio`, `preprocessed_X`, `load_table`), with no new test logic.
- Tests: `D:\Orange\python.exe -m pytest tests -q -p no:warnings` → 47 passed, 0
  skipped. bank-full is read from `../magang-datamining/datasets/bank-full.csv` or the
  `ZEST_BANK_FULL` environment variable; if it is missing, the bank tests are skipped.
- **GitHub Actions for the PCA tests** (`3ae50a7`). `.github/workflows/tests.yml`:
  ubuntu-latest, Python 3.12, package versions the same as Orange's bundled Python, Qt
  `offscreen`. First run on main green: 40 passed, 7 skipped (all bank-full tests, the
  data is not in the repo).

Maximum difference vs Orange PCA (widget projector, Normalize ON, `n_components=100`).
All pass the test rules (non-null rtol=1e-5 atol=1e-8; null space rtol=0 atol=1e-8):

| Dataset | Compared | components | scores | null scores | expl. var | ratio |
|---|---|---|---|---|---|---|
| iris (150×4, null 0) | zestcore | 0.0 | 5.55e-17 | – | 0.0 | 0.0 |
| | scikit-learn | 0.0 | 0.0 | – | 0.0 | 0.0 |
| | GUI | 0.0 | 0.0 | – | – | 4.79e-7 (CSV %g) |
| heart_disease (303×25, null 7) | zestcore | 0.0 | 8.88e-16 | 4.44e-16 | 0.0 | 0.0 |
| | scikit-learn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | GUI | 0.0 | 0.0 | 0.0 | – | 3.58e-7 (CSV %g) |
| bank-full (45211×51, null 9) | zestcore | 0.0 | 3.11e-15 | 1.33e-15 | 0.0 | 0.0 |
| | scikit-learn | 0.0 | 1.78e-15 | 8.88e-16 | 0.0 | 0.0 |
| | GUI | 0.0 | 0.0 | 0.0 | – | 4.85e-8 (CSV %g) |

Tied components: heart_disease PC2, 3; bank-full PC2, 4, 6, 7 (s = +1 for every
reference). The number of zero directions from the formula = measured for all three.

Decisions approved by Raditya (do not change without discussion):

- Covariance in exactly sklearn's form and multiplication order (see step 5).
- Scores in the sklearn form X@C.T − mean@C.T (`_base.py:149, 156`). zestcore
  components stay C-order. The score difference zestcore vs sklearn (≤1.8e-15) comes
  from the memory layout of the components (sklearn F-order), which makes BLAS take a
  different path; with an F-order copy the difference is 0.0 (evidence cell in notebook
  02). This is a BLAS detail, not a method, so it is not reproduced.
- Tests keep using the spec tolerance and do not assert bit-identity (depends on BLAS).

### Open questions

- bank-full: the Orange model scores (`model(table)`, the `Table.transform` path) differ
  by 1.78e-15 from sklearn's `proj.transform(X)` on the same X (iris, heart_disease:
  0.0). This is the source of the scikit-learn vs Orange score difference in the table.
  Not investigated yet; far below the tolerance.
- bank-full, Seam A (T2a): `zestcore.transform` differs by 3.55e-15 from `model(test).X`
  (iris, heart_disease: 0.0). Probably the same source as the item above. Not
  investigated yet; far below the tolerance.

### Next tasks

1. ~~**T2a: Zest PCA widget**~~ done 27 Nov 2026 (see the T2a section above). Recorded
   follow-ups: scree plot, clearer error messages, and the `nan` ratio on constant data.
2. **ExtDT widget** (extended decision tree), the first Stage 2 model, Prof. Irfan's
   direction of 23 Nov 2026. Uses the T2a template (three-layer pattern, ADR 0003).

### Further steps (outside Stage 1)

- **bank-additional-full** (41188×20 → 41188×63 after Continuize). Only its shape and
  zero directions have been checked (11 vs a prediction of 10, see Test rules, rule 2);
  not yet compared with zestcore, scikit-learn, or the GUI. Stage 1 is complete with
  iris, heart_disease, and bank-full.
