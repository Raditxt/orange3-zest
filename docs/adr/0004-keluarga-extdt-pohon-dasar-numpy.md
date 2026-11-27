---
status: proposed
---

# ExtDT model family on top of a NumPy base tree

## Context

Prof. Irfan's direction (23 Nov 2026): the first Stage 2 model is the "extended decision
tree" (ExtDT). The T2b review found that this term does not point to a single algorithm
but to several candidates. The reference, Orange 3.40.0's Tree, has already been
dissected in `docs/orange-tree-internals.md` (Claims 1-7). This ADR builds on ADR 0002
(preprocessing through Orange preprocessors) and ADR 0003 (three-layer pattern:
zestcore, then an Orange adapter, then a thin widget; for ExtDT the adapter is a
`Learner`/`Model` pair).

Facts that bind all the options below. Paths relative to
`D:\Orange\Lib\site-packages\`.

- **Orange's Tree is not sklearn.** The Tree widget uses Orange's `TreeLearner`:
  information gain in bits, threshold = the largest data value on the left side, NaN is
  not imputed, and discrete attributes are handled directly (Claims 1-5).
- **sklearn `ExtraTreesClassifier` (1.5.2).**
  - The threshold is drawn uniformly at random in [min, max] of the feature at the node
    (`sklearn/tree/_splitter.pyx:785-792`).
  - The criterion can only be `gini`, `entropy`, or `log_loss`
    (`sklearn/tree/_classes.py:941`). The normalized score 2I/(Hs+Hc) from Geurts et al.
    (2006) is not available. A new criterion can only be added as a Cython `Criterion`
    subclass, not in NumPy.
  - NaN is rejected, because `allow_nan` only applies to `splitter="best"`
    (`_classes.py:1085-1090`). Checked: fitting with one NaN gives `ValueError: Input X
    contains NaN`.
- **Orange's pattern for wrapping sklearn.** `RandomForestLearner` is a `SklLearner`
  (`Orange/classification/random_forest.py:42-44`) with the default preprocessors
  `HasClass, Continuize, RemoveNaNColumns, SklImpute` (`Orange/base.py:549-553`).
  Discrete attributes are one-hot encoded and NaNs imputed before reaching sklearn.
- **Preprocessors from the widget replace the default preprocessors.** The rule is in
  `Orange/base.py:53-58`. The exception is a `Fitter` learner, which is given
  `use_default_preprocessors = True` (`Orange/widgets/utils/owlearnerwidget.py:203-204`).
  For non-Fitter learners, the widget only shows the info message "Ignoring default
  preprocessing" (`owlearnerwidget.py:87-91, 216-219`).
- **Finding T2b-1 (Claim 1).** In a `Fitter`, the `preprocessors` kwarg is dropped on the
  regression path, because the regression `__init__` has no parameter with that name
  (`Orange/modelling/base.py:82-86`).

## Decision

Raditya decided (26 Nov 2026; translated from Indonesian):

> ExtDT is built as a family of models on top of one NumPy base tree, in this order:
> Extra Trees (pilot, Geurts 2006, compared against sklearn `ExtraTreesClassifier`)
> → multi-label DT → C4.5 (compared against Weka J48) → UDT → Extension DT (preceded by
> further research: find a public dataset + reference numbers).

The five questions below are **not yet decided**. Each question lists options with
their consequences, and the choice is left to Raditya.

Terminology note: "ExtDT" is now used as the family name, while "Extension DT" is one of
its members, and "Extra Trees" is another member. The three are easy to mix up in the
report. Canonical names need to be fixed before the first code.

## Question 1. Write from scratch (NumPy) or wrap sklearn

### 1a. Base tree written from scratch in zestcore (pure NumPy)

- Consistent with CLAUDE.md and ADR 0003: zestcore holds the algorithm, the adapter only
  wraps.
- The 2I/(Hs+Hc) score can be written directly, as can C4.5 gain ratio, multi-label
  entropy, and UDT fractional weights. All family members share the same code.
- Pilot results will not be identical to sklearn, because the random number generator
  and the order of attribute sampling differ. Only distributional equivalence can be
  claimed (Question 3).
- Speed risk: node building runs in Python. Extra Trees is relatively light (one random
  threshold per candidate attribute, can be vectorized per node), but a 100-tree
  ensemble on bank-full (45211 rows) has never been measured. The time must be measured,
  not assumed.

### 1b. Wrap sklearn `ExtraTreesClassifier` like `RandomForestLearner`

- Least code (the `SklLearner` pattern). For the same seed, results are identical to
  sklearn, and fast.
- There is no code in zestcore, so the rule "orangecontrib only wraps zestcore" is
  broken, or a written exception is needed.
- The paper's 2I/(Hs+Hc) score cannot be used without writing a Cython `Criterion`. As a
  result the pilot becomes "sklearn's Extra Trees", not the Geurts 2006 version.
- Nothing can be reused by C4.5, UDT, or multi-label. The "one base tree" decision only
  takes effect from the second member, so the pilot does not test the base tree at all.

### 1c. NumPy base tree with a selectable score: `entropy` and `paper`

Same as 1a, plus an `entropy` score mode (plain information gain, like
`criterion="entropy"` in sklearn) next to the paper's 2I/(Hs+Hc) score.

- The comparison with sklearn can be split into two steps:
  1. `entropy` mode vs sklearn: differs only in random numbers.
  2. `paper` mode vs `entropy` mode: differs only in the score function.
  In 1a, these two differences are mixed into one number.
- The cost is one extra score function with its tests. In the widget, this mode can be
  hidden.

## Question 2. Base tree boundary and extension points

The goal is that E/D/C (the next members) only *add* code, not *change* the base tree.
The four extension points from the brief (split criterion, threshold selector, stopping
rule, ensemble aggregation) turn out not to be enough. The table below maps what each
member needs. The source is general knowledge of these algorithms and has not been
rechecked against each paper (see Status).

| Need | Extra Trees | Multi-label | C4.5 | UDT | Extension DT |
|---|---|---|---|---|---|
| Split criterion | 2I/(Hs+Hc) | multi-label entropy | gain ratio | weighted entropy | ? |
| Threshold selector | random in node [min, max] | exhaustive | data value (≤ midpoint) | exhaustive over pdf points | ? |
| Attribute selection | K random | all | all | all | ? |
| Target y shape | 1 class column | **n × L matrix** | 1 column | 1 column | ? |
| Row weights | no | no | **yes** (fractional missing) | **yes** (fractional tuples) | ? |
| Multiway discrete split | optional | optional | **yes** | ? | ? |
| Post-build | no | no | **pruning** | ? | ? |
| Ensemble aggregation | yes | no | no | no | ? |

What the four points from the brief do not cover is marked in bold:

- **Target shape.** Multi-label changes the leaf from "count per class" to "frequency
  per label". If the base tree assumes a 1-D y, the second member already has to change
  the base tree.
- **Row weights.** C4.5 and UDT split one row across several branches with fractional
  weights. If weights are not there from the start, the third member has to change the
  base tree.
- **Pruning.** This is a step after the tree is built, not part of the stopping rule.

### 2a. Strategy functions (callable parameters)

`build_tree(X, y, w, *, score, propose_splits, should_stop, rng)` in zestcore. Each
family member is a set of functions. Ensembling and pruning are separate functions that
call or take a tree.

- Pure NumPy, and each function can be tested on its own against brute-force numbers,
  as in `tree_internals_check.py`.
- Consistency between functions is not enforced by the structure. For example, a
  multi-label score given a 1-D y only fails at run time, so upfront validation is
  needed.

### 2b. Base class with overridden methods

`BaseTree` has `_score`, `_propose_splits`, `_should_stop`, and `_leaf_value`. Each
family member is a subclass.

- Easier to read for readers used to OOP, and fits Orange's `Learner`/`Model` pair.
- There is a risk of inheritance coupling. Subclasses often end up depending on the
  internal call order, so a change in `BaseTree` can silently break other members.

### 2c. Data contract only, no shared framework

The base tree only defines the node structure (arrays `feature`, `threshold`,
`children`, `value`, `weight`) and the prediction function. Each member writes its own
builder.

- Simplest for the pilot, and the node contract is still shared.
- The recursion logic and stopping rules are duplicated in every member. This departs
  from the intent of the "one base tree" decision.

Consequence for all options: if a 2-D `y` and `w` are not in the contract from the
pilot, the "add, not change" promise does not hold for multi-label or C4.5. These two
add no meaningful pilot code (y can be `(n, 1)` and w can be all 1), but they must be
tested from the start.

## Question 3. Pilot correctness claim

Pilot results cannot be identical to sklearn (Question 1). Per CLAUDE.md, that is a
valid finding and its cause must be written down. There are three layers of claims,
which can be taken partly or all together.

### 3a. Deterministic components: identical

This covers the score function vs brute-force computation, the threshold always lying
in the node's [min, max], the stopping rule, and predictions of a hand-built tree.
Results are tested with `np.allclose` and the maximum difference is reported, as in
Stage 1. This claim is strong and cheap.

### 3b. Ensemble: distributionally equivalent to sklearn

- Matched settings: `n_estimators`, `max_features` = K, `min_samples_split` = n_min,
  `bootstrap=False`, `criterion="entropy"` (compared with the `entropy` mode in 1c).
- Experiment design: R seeds × repeated stratified k-fold CV (for example 10 × 10), then
  mean ± sd of accuracy/AUC per dataset.
- **The "equivalent" criterion must be written before looking at the numbers.** For
  example: mean difference < x, or a paired per-fold test with a given threshold. Setting
  the criterion after seeing the numbers amounts to covering a difference with a loose
  tolerance.
- Known differences that must be recorded:
  - The RNG and the order of attribute sampling differ.
  - sklearn skips constant attributes when drawing K attributes.
  - Aggregation differs. sklearn averages probabilities, while Geurts 2006 (as far as I
    remember) uses majority voting for classification. This needs to be checked in the
    paper.

### 3c. Performance vs Orange's Tree and Random Forest

Comparison through Test and Score with the same folds, plus `.ows` evidence. This is not
a correctness claim but a usefulness claim. The reference settings must be stated:

- Tree uses the widget defaults (`binarize=True`, etc., Claim 1).
- RF uses Continuize + SklImpute.

Consequence: 3a + 3b prove "the implementation is correct", 3c answers "is it useful".
Without 3b, the pilot claim is only "plausible". Without 3a, a failure in 3b is hard to
trace to its cause.

## Question 4. Missing values and discrete attributes: in the tree or in preprocessing

ADR 0002 sets preprocessing through Orange preprocessors, with zestcore only taking
ready-to-use numeric X, "unless a model genuinely needs special preprocessing". Orange's
Tree itself falls under that exception, since it has no preprocessors (Claims 1, 3, 5).

### 4a. Handled in the tree

zestcore takes X with NaN, plus simple metadata as plain NumPy arrays: `is_discrete` and
`n_values` per column.

- This is the only way C4.5 can be compared with J48, which has multiway discrete
  splits and fractional missing handling internally. UDT also needs a special
  representation.
- This becomes the first exception to ADR 0002, and it must be written down. zestcore
  starts carrying column metadata, which ADR 0002 option 1 avoided.
- The adapter still uses `HasClass` (rows with a NaN target are dropped). This is the
  same as Orange, Claim 5a.

### 4b. Orange preprocessing (the `SklLearner` pattern: Continuize + Impute)

- Fully consistent with ADR 0002, and zestcore stays smallest.
- The pilot comparison with sklearn is fair, because sklearn also takes one-hot,
  already-imputed data.
- One-hot changes Extra Trees' behavior. The K random attributes are drawn from the
  indicator columns, not from the original attributes, so discrete attributes with many
  values are picked more often. This departs from Geurts, which handles categorical
  attributes directly.
- C4.5 and UDT will still need 4a later, so the zestcore input contract changes midway.

### 4c. Staged: the pilot uses 4b, the input contract already has slots for 4a

`build_tree` takes `is_discrete`/`n_values` from the start (pilot: all continuous) and
allows NaN in the contract. The pilot itself rejects NaN with a clear message.

- The pilot stays simple, and later members only add without changing function
  signatures.
- Some parameters are unused in the pilot, and that has to be explained when defending
  the code.

Consequence for all options: if the adapter has default preprocessors and the user
connects a Preprocess widget, the default preprocessors are **replaced**, not added to
(`Orange/base.py:53-58`). Under option 4b, this means NaN can reach zestcore through the
GUI. Input validation in the adapter or in zestcore is required under every option.

## Question 5. Preprocessors lost on the regression path

Finding T2b-1: a `Fitter` drops the `preprocessors` kwarg if the regression learner's
`__init__` does not name it. For ExtDT, this means preprocessors from the widget are
lost with no message at all if the adapter uses a `Fitter` and its regression learner
forgets to name that parameter.

### 5a. Classification-only pilot, no `Fitter`

`ZestExtraTreesLearner` is directly a classification `Learner`.

- The problem does not arise in the pilot. Extra Trees does have a regression variant,
  but the sklearn reference is classification.
- The problem is only postponed. As soon as regression is added, Question 5 comes back.
  Non-Fitter learners also do not get `use_default_preprocessors = True` from the widget
  (Question 4).

### 5b. Own `Fitter`, both learners must accept `preprocessors` + a marker test

Both `__init__` methods name `preprocessors` explicitly. The test mirrors
`check_klaim1`: the marker must reach the classification learner **and** the regression
learner.

- One widget can serve both target types, like the Tree widget, and the default
  preprocessors stay active because of the Fitter (`owlearnerwidget.py:203-204`).
- More code, and the marker test must be maintained for every family member.

### 5c. No Preprocessor input from the widget

Preprocessors are fixed in the learner class, and the widget does not forward a
Preprocessor input.

- The problem disappears entirely and the behavior is easy to explain.
- Unlike every Orange learner widget, which has a Preprocessor input
  (`owlearnerwidget.py:92`). Users used to connecting Preprocess will be surprised that
  the input is missing.

Consequence for all options: the adapter shape chosen becomes the template for every
family member, so the choice should be made before the pilot. The marker test (5b) can
be added under any option and is cheap.

## Status

### Handoff 26 Nov 2026 (T2b-2)

**Done**

- Draft of this ADR (status `proposed`), with Raditya's family-order decision and five
  questions with their options. No zestcore code.
- The sklearn 1.5.2 and Orange 3.40.0 facts in the Context section were checked against
  the installed code. The NaN fact for `ExtraTreesClassifier` was checked by running a
  fit.
- Notion: the T2b task notes were filled in and its status set to Done. A 📊 Report was
  sent to the PM.

**Unverified / needs attention**

- The paper claims in this ADR were originally written from general knowledge, because
  the research file did not exist yet when the draft was made. The cross-check against
  `docs/research/extdt-kandidat-rujukan.md` (commit e312660) is in the section "Paper
  claim cross-check against the research file" below. Options and recommendations were
  not changed.
- Canonical names: ExtDT, Extension DT, Extra Trees (see the terminology note). There is
  no `GLOSSARY.md` yet. It will be created once the names are fixed.

### Paper claim cross-check against the research file (26 Nov 2026)

Label meanings:

- **matches**: the content is the same as in the research file, and the research file
  does not mark it [unverified].
- **differs**: the research file states something else.
- **unverified**: the research file does not discuss it, or marks it [unverified].

The research file itself is not a primary source, so the label "matches" means
consistent with the research file, not checked directly against the PDF.

| No | Claim in ADR 0004 (location) | Research file content | Status |
|---|---|---|---|
| 1 | Extra Trees score in Geurts 2006 = 2I/(Hs+Hc), not in sklearn (Context, 1b, 1c, Q2 table) | Section A: the paper uses normalized information gain 2·I/(H_s+H_c), sklearn gini/entropy/log_loss | **matches** |
| 2 | Geurts 2006 uses majority voting, sklearn averages probabilities (3b) | Not discussed at all | **unverified**: still needs checking against the Geurts PDF |
| 3 | C4.5 threshold = largest data value ≤ midpoint (Q2 table) | Section D only mentions gain ratio, fractional missing, pruning CF=0.25, subtree raising, and the MDL correction. The threshold rule is not mentioned | **unverified**. What **matches** the Q2 table: gain ratio, weights for fractional missing, and pruning |
| 4 | UDT needs row weights (fractional tuples), weighted entropy, exhaustive threshold over pdf points (Q2 table) | Section C: UDT searches splits over pdf intervals (s=100 sample points). `sample_weight` expansion is called "only an approximation, not UDT-style split search (inference)". Fractional tuples are not mentioned | **partly differs**: split search over the pdf matches, but the research file states that row weights alone are not enough for UDT. Fractional tuples **unverified** |
| 5 | Multi-label: multi-label entropy (Clare & King 2001), target n × L, leaf = frequency per label (Q2 table, Q2 point "Target shape") | Section E: entropy −Σ[p log p + q log q] and the leaf holds a **set of labels**. Gain vs gain ratio and the leaf labeling rule **[unverified]** | Entropy **matches**. Leaf content **differs**: the paper says a set of labels, the ADR says frequency per label. Leaf labeling rule **unverified** |

Other findings from the research file that touch the ADR text. They are only recorded
here; the option text is not changed.

- **1b** states that wrapping sklearn provides nothing reusable for multi-label.
  According to section E, `DecisionTreeClassifier` with a 2-D y and
  `criterion="entropy"` gives an impurity equivalent to ML-C4.5 entropy divided by L.
  This means sklearn can still serve as a partial reference for multi-label.
  - The 298/300 test in the research file was run with sklearn 1.9.1, not 1.5.2.
  - That test is also unpublished.
- **Q2 table, Extra Trees, "multiway discrete split: optional".** According to section
  A, the paper uses random subset splits (two branches) for nominal attributes, not
  multiway.
- **Q2 table, Extension DT, all "?".** Section B: a matter-element model, discretization
  with a dependent function, then an ID3 tree (information gain), then an extension
  transform on the rules. No code, public dataset, or accuracy numbers.
- **Rounding of K.** sklearn uses floor, the paper says "rounded" (section A, marked
  [unverified]). This matters for 3b, because `max_features` has to be matched.
- **Multi-target in the adapter.** Section E: `TreeLearner`/`SklTreeLearner` do not set
  `supports_multiclass`, so multi-target data triggers a `TypeError`. A multi-label
  adapter needs `supports_multiclass = True`.

**Next steps (proposed order, schedule set by the PM)**

1. Raditya picks options for Questions 1-5, writes them in the Decision section, and
   changes the status to `accepted`.
2. Check the still-unverified claims against the PDFs: Geurts aggregation (no. 2), C4.5
   threshold (no. 3), UDT fractional tuples (no. 4), and ML-C4.5 leaf content (no. 5).
3. Write the "equivalent" criterion (3b) before the experiment.
4. Break the pilot into implementation tasks: a small score-function example on iris
   (3a), then a single tree, the ensemble, the adapter, and the widget.
