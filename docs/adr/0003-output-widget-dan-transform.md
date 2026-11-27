---
status: accepted
---

# Zest PCA widget outputs and transforming new data

## Context

Task T2a (Prof. Irfan's direction, 23 Nov 2026): the Zest PCA widget wraps
`zestcore.pca`, outputs at least Transformed Data and Components, and can project new
data. The widget becomes the template for the next model (ExtDT). Preprocessing uses
Orange preprocessors (ADR 0002).

How OWPCA does it (Orange 3.40.0, paths relative to `Orange/`):

- The Transformed Data domain is built in `DomainProjection.__init__`
  (`projection/base.py:190-211`). Each component is a `ContinuousVariable` with
  `compute_value=ComputeValueProjector(feature=i, transform=TransformDomain(self))`.
- `TransformDomain.__call__` converts the data to `pre_domain` (the preprocessed domain,
  stored at `projection/base.py:65,68`), then calls `proj.transform(X)`
  (`projection/base.py:139-142`, `115-116`).
  `ComputeValueProjector.compute` only takes the i-th column (`projection/base.py:175-176`).
  The `TransformDomain` result is cached per source table, so the projection is computed
  once for all components (`data/table.py:312-317`).
- Apply Domain only runs `data.transform(template.domain)`
  (`widgets/data/owtransform.py:27`). The `pre_domain` variables carry the
  preprocessors' compute_value with training-data statistics, for example Normalize
  stores the training SD (`preprocess/normalize.py:47-67`). New data is automatically
  processed with training statistics.
- Training scores also go through this path: `self._pca(self.data)` (`owpca.py:321`)
  calls `data.transform(self.domain)` (`projection/base.py:118-119`). Scores are not
  taken from the fit result.
- The Components output deliberately has no transform: `compute_value=lambda _: None`
  (`owpca.py:336-355`).
- Variables without compute_value that are not in the source table are filled with
  missing values (`data/table.py:307-310`).

`zestcore.pca` (`zestcore/pca.py:17`) currently only returns scores for the X used in
the fit. There is no function yet to project new X.

## Options

Options 1a-1c all produce Transformed Data that can be used by Apply Domain. They differ
in who builds the domain and runs preprocessing.

### 1a. Adapter handed to Orange's `DomainProjection`

An adapter class in `orangecontrib/zest/` stores `components_` and `mean_`, with a
`transform(X)` that calls zestcore. The widget runs the preprocessors itself, then
creates `DomainProjection(adaptor, pre_domain, k)` and sets `pre_domain` and `name`
manually. Those two attributes are normally set by `Projector.__call__`
(`projection/base.py:65-69`).

- Amount of code: least, about 15-20 lines of adapter plus preprocessing in the widget.
- Dependencies: `DomainProjection` is in `__all__` (`projection/base.py:21-22`), but its
  contract is implicit. The adapter must have the attributes read through
  `__dict__.update(proj.__dict__)` (`projection/base.py:112`, `201-202`). `pre_domain`
  is set from outside, whereas in Orange that attribute is set by `Projector`.
- Training scores through compute_value: yes, if the widget calls `model(data)` as in
  `owpca.py:321`.
- Pickle: Transformed Data saved as `.pkl` carries the adapter in compute_value. The
  adapter class must be defined at module level, and the file can only be opened when
  orange3-zest is installed. Save Model is not relevant because its input is
  `Orange.base.Model` (`widgets/model/owsavemodel.py:22`), and a projection is not a
  `Model`. This applies to all options.

### 1b. Own compute_value following the `TransformDomain`/`ComputeValueProjector` pattern

`orangecontrib/zest/` writes a shared compute_value class (subclass of
`SharedComputeValue`) and a per-column one, then builds its own `Domain`. The
`TransformDomain` here also does the conversion to `pre_domain`.

- Amount of code: most, about 50-70 lines. Includes `__eq__`/`__hash__`, which are
  required for correct table caching (`data/util.py:77-84`, `projection/base.py:144-162`).
- Dependencies: only `SharedComputeValue`, `Domain`, and `ContinuousVariable`, which are
  public data API. No dependency on `Orange.projection`.
- Training scores through compute_value: yes, if the widget calls
  `data.transform(domain)`.
- Pickle: same as 1a. The compute_value classes must be at module level and need
  orange3-zest installed. Local functions or lambdas cannot be pickled.

### 1c. `ZestPCA(Projector)` + `ZestPCAModel(DomainProjection)`

This pair mirrors `PCA`/`PCAModel` (`projection/pca.py:27-53`, `73-78`) and
`LinearProjector` (`projection/base.py:241-259`). `ZestPCA.preprocessors` contains
Continuize → SklImpute (→ Normalize if checked). `fit(X)` calls `zestcore.pca` and
returns a `ZestPCAModel`. `Projector.__call__` runs preprocessing and sets `pre_domain`
(`projection/base.py:54-71`).

- Amount of code: small, about 30-40 lines, and the widget stays thin.
- Dependencies: `Projector` and `DomainProjection` (both in `__all__`), used exactly
  following Orange's own subclass pattern. The `proj.transform`/`components_` and
  `mean_` contract stays implicit as in 1a.
- Training scores through compute_value: yes, the same path as OWPCA (`model(data)`).
- Pickle: same as 1a. `Projector` already handles its thread-local state
  (`projection/base.py:99-106`). `ZestPCA` can also be sent as an output equivalent to
  OWPCA's `PCA` output (`owpca.py:364`).

### 2. Plain NumPy table without transform

The widget calls `zestcore.pca`, then creates a `Table` from `result.scores` with PC
variables that have no compute_value.

- Amount of code: small, about 15 lines in the widget, and no transform function is
  needed in zestcore.
- Dependencies: only `Table`, `Domain`, and `ContinuousVariable`.
- Training scores through compute_value: no. Scores are taken directly from
  `result.scores`, so the path differs from OWPCA.
- Pickle: the `.pkl` can be opened without orange3-zest. New data cannot be projected:
  Apply Domain produces empty PC columns (`data/table.py:307-310`). The T2a requirement
  ("transform for new data") is not met.

## Consequences

- **zestcore needs `transform(X, mean, components)` for options 1a-1c.** The formula
  must live in zestcore. If it were written in the widget layer, the "only wraps" rule
  in CLAUDE.md would be broken. The form of the formula determines how close the result
  is to sklearn:
  - sklearn computes `X @ C.T - mean @ C.T` (`sklearn/decomposition/_base.py:148-155`).
  - `(X - mean) @ C.T` is mathematically equivalent, but the floating-point order
    differs. The result is only identical within the spec tolerance, not bit-identical.
  - `zestcore.pca` already uses the sklearn form for training scores
    (`zestcore/pca.py:64-65`). `transform` should use the same form, and `pca` should
    call it, so there is a single source for the formula.
  - Whether the result is bit-identical to Orange still has to be measured (maximum
    difference reported), because `mean` and `components` themselves come from
    zestcore's computation path.
- **Template for ExtDT: the three-layer pattern.**
  1. `zestcore`: pure-NumPy algorithm.
  2. Orange adapter classes in `orangecontrib/zest/`: `Projector`/`DomainProjection` for
     projection, `Learner`/`Model` for ExtDT.
  3. A thin widget that only manages settings, calls the adapter, and sends outputs.

  For a learner, preprocessing and storing `original_domain` are done by
  `Learner.__call__` (`Orange/base.py:124-162`). New data is converted to the model
  domain by `Model.data_to_model_domain` (`Orange/base.py:343-386`) and comes in through
  the Predictions widget. The per-column compute_value mechanism (`DomainProjection`,
  `TransformDomain`, `ComputeValueProjector`) is not used in ExtDT. What carries over to
  ExtDT is the three-layer pattern, preprocessing through Orange preprocessors (ADR
  0002), and the widget skeleton. Of the four options, 1c is the one whose shape is most
  parallel to the `Learner`/`Model` pair. 1a, 1b, and 2 have no such pair.
- **Evidence of identity in tests and `.ows`:** with options 1a-1c, scores are compared
  through the same path as OWPCA (`model(data)`). With option 2, the tests compare
  `result.scores` with OWPCA scores from the compute_value path.

## Decision

I chose option 1c: ZestPCA(Projector) and ZestPCAModel(DomainProjection) in
orangecontrib/zest.

- fit() calls zestcore.pca on X that has gone through the Projector's preprocessors
  (Continuize → SklImpute → Normalize if checked). zestcore stays pure NumPy.
- zestcore gets a transform(X, mean, components) function with the same formula as
  zestcore.pca and sklearn: X @ C.T − mean @ C.T.
- Transformed Data scores are computed through compute_value, both for training data
  and for new data via Apply Domain, just like OWPCA (owpca.py:321).
- All classes stored in the output domain (the model and the proj object) are defined at
  module level so they can be pickled, and this is tested. The form of the proj object
  (ZestPCA itself like LinearProjector, or a separate small object) is decided during
  implementation with a small example.
- Claim: Zest PCA results are identical to Orange PCA within the spec tolerance.
  Bit-identity is only recorded as an observation.

Reason: the path is the same as OWPCA, so the comparison is like for like.
Preprocessing and training-data statistics carry over to new data with no extra code
(ADR 0002). The three-layer pattern (zestcore → Orange adapter classes → thin widget)
becomes the template for ExtDT, with the Learner/Model pair replacing
Projector/DomainProjection.

## Implementation notes

Form of the proj object (25 Nov 2026, T2a step 4a): **a separate small object**
`ZestPCAFit`, created fresh in every `ZestPCA.fit()`. It carries `mean_`,
`components_`, `explained_variance_`, and `explained_variance_ratio_`. The ratio is
needed by the widget for `var.attributes["variance"]` (`owpca.py:325-328`).
`transform(X)` calls `zestcore.transform`. The pattern is the same as Orange PCA, which
wraps a fresh sklearn object (`projection/pca.py:51-53`).

The alternative, `ZestPCA` itself as proj (the `LinearProjector` pattern,
`projection/base.py:251-259`), was rejected based on a small example on iris
(`docs/proofs/proj_forms_check.py`):

- An old model changes when the same projector is refitted. `DomainProjection` copies
  `proj.__dict__` (`projection/base.py:112`), but `transform` still calls
  `self.proj.transform` (`:115-116`). After the projector is refitted on `iris[:50]`,
  `model(iris)` from the first fit changes with a max difference of 5.46, and
  `model.components_` no longer equals `model.proj.components_`.
- Pickling the output domain or the model fails (`cannot pickle '_thread._local'`). The
  `__dict__` copy also carries `_Projector__tls` into the model, and that attribute is
  only dropped by `Projector.__getstate__` (`:99-102`), not by `DomainProjection`. With a
  separate object, the iris output domain and model can be pickled (2807 bytes).

Template rule: **`transform(X)` only uses statistics from the fit, never statistics of
the X being transformed.** `Table.transform`/`from_table` run compute_value in chunks of
at most 5000 rows (`_FromTableConversion.max_rows_at_once`, `data/table.py:385`,
`423-448`). `compute_shared` (here `TransformDomain`) receives that chunk of rows
(`data/table.py:303, 316`), not the whole table, and the cache is cleared after each
chunk (`:446-448`). Evidence: `model(table)` on bank-full (45211 rows) calls
`ZestPCAFit.transform` 10 times (9 × 5000 + 211 rows), whereas iris needs only one call.
The 4a sensitivity test shows the consequence: if `transform` uses `X.mean(axis=0)`, the
bank-full training scores also come out wrong, even though the table is the same as the
fit data, because each chunk uses its own mean. iris and heart_disease still pass
because they are a single chunk. This rule also applies to the next model (ExtDT) and to
custom preprocessors.
