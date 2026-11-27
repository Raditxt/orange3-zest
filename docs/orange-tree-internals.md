# Specification: how Orange 3.40.0 builds a Tree (ExtDT reference)

**Status: DRAFT for Raditya's review (26 Nov 2026), task T2b-1.** There is no zestcore
code for trees yet; this document only dissects the reference.

Environment: `D:\Orange\python.exe` (Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2,
NumPy 2.1.3). Every `file:line` refers to the code installed in
`D:\Orange\Lib\site-packages\Orange\`, not GitHub master. Proof script:
`docs/proofs/tree_internals_check.py` (one `check_klaimN()` function per claim, using
`assert`, exit 0 = all claims proven). Data: iris (150 rows, 4 numeric, 3 classes) and
heart_disease (303 rows, 13 mixed attributes, NaN in `major vessels colored` (4) and
`thal` (2)). For regression, heart_disease is used with `max HR` as the target.

Unless stated otherwise, all numbers use the **Tree widget defaults**:
`binarize=True, min_samples_leaf=2, min_samples_split=5, max_depth=100,
sufficient_majority=0.95`.

Note: the core scorer (`Orange/classification/_tree_scorers.*.pyd`) is compiled Cython
and its `.pyx` source is not installed. Claims about its contents (score formula,
threshold candidates) are proven behaviorally: the results are matched against a
brute-force NumPy recomputation in the proof script, not by reading the code.

## Claims

### Claim 1. The Tree widget uses Orange's TreeLearner, not sklearn

- `widgets/model/owtree.py:62` `LEARNER = TreeLearner` from `Orange.modelling.tree`,
  which is a `Fitter` (`modelling/tree.py:22-28`) that picks a learner by target type
  (`modelling/base.py:69-80`): a discrete target goes to
  `Orange.classification.tree.TreeLearner`, a numeric target to
  `Orange.regression.tree.TreeLearner`. `SklTreeLearner` (a wrapper of
  `sklearn.tree.DecisionTreeClassifier`, `classification/tree.py:234-247`) is **not**
  used by the widget.
- The output model is `Orange.tree.TreeModel` for both.
- Mapping of widget settings to learner parameters (`owtree.py:107-117`):

  | Widget setting (default) | Learner parameter | When the checkbox is off |
  |---|---|---|
  | Induce binary tree (ON) | `binarize=True` | `False` |
  | Min. instances in leaves (2) | `min_samples_leaf=2` | `1` |
  | Do not split subsets smaller than (5) | `min_samples_split=5` | `2` |
  | Limit maximal tree depth (100) | `max_depth=100` | `None` |
  | Stop when majority reaches (95%) | `sufficient_majority=0.95` | `1` |

  Note: the learner's default `binarize` is `False` (`classification/tree.py:63`), but
  the widget sends `True`. The ExtDT reference must state which setting is used.
- The tree learners have no default preprocessors (`preprocessors = ()` for both): no
  imputation, continuize, or normalization. Discrete attributes and NaN are handled
  directly by the algorithm (Claims 3, 5).
- **Finding:** preprocessors from the widget's input signal only reach the
  classification learner. `Fitter.__kwargs` filters kwargs by
  `__init__.__code__.co_varnames` (`modelling/base.py:82-86`), and the regression
  `__init__` (`regression/tree.py:57-60`) has no `preprocessors` name, so that kwarg is
  dropped, and so is `sufficient_majority`. Evidence: the marker `["PENANDA"]` shows up
  in the classification learner, not in the regression one. Impact on ExtDT: if the
  regression reference uses a Preprocess widget, Orange's regression path ignores it.
  Not yet tested in the GUI.

### Claim 2. Split criterion: information gain (classification), MSE reduction (regression)

- Classification: **information gain with base-2 entropy** (bits), not Gini and not
  gain ratio. For multiway attributes the formula is written directly in
  `classification/tree.py:114-117`; for numeric attributes and binarization it goes
  through `find_threshold_entropy` / `find_binarization_entropy` (Cython).
- Regression: **MSE reduction** = var(y) − Σ (nᵢ/n)·var(yᵢ), var with ddof=0
  (`find_threshold_MSE`, `find_binarization_MSE`, `compute_grouped_MSE`).
- Attribute selection: each attribute's best score is compared with `sc > best_score`
  (`classification/tree.py:178`, `regression/tree.py:138`), so **on a tie, the
  attribute with the smallest index wins**. Attributes with a score of 0 are rejected.
- Evidence (brute-force NumPy over all split candidates, then matched against the root
  of Orange's tree):

  | Data | Orange root | Brute-force root | Score |
  |---|---|---|---|
  | iris | petal length | petal length | 0.9182958341 bit |
  | heart_disease | thal | thal | 0.2063531160 bit |
  | heart_disease, target max HR | slope peak exc ST | slope peak exc ST | 104.1088319516 |

  On iris, `petal length` and `petal width` are tied (0.918 bit); `petal length` is
  chosen because its index is smaller. Cython score vs NumPy formula: iris
  `sepal length` difference 2.2e-16; regression `age` difference 1.1e-12.

### Claim 3. Discrete attributes: multiway or optimal 2-group

- `binarize=False`: one branch per **declared value** (`DiscreteNode`,
  `classification/tree.py:87-124`). heart_disease: root `thal` with 3 branches. Values
  with a count < `min_samples_leaf` are emptied; their branch becomes `None`
  (`classification/tree.py:101-106, 190-191`).
- `binarize=True` and an attribute with > 2 values: **exhaustive search** over all
  2^(k−1) − 1 groupings of the values into two branches, in Gray code order
  (`find_binarization_entropy`, `find_binarization_MSE`), with the `min_samples_leaf`
  limit per group. The result is a `MappedDiscreteNode` with a value→branch `mapping`.
  heart_disease: `thal` → mapping `[1, 0, 0]` (normal to the right, the rest to the
  left), the same as brute force.
- A 2-valued attribute stays a `DiscreteNode` even with `binarize=True`
  (`classification/tree.py:129-130`, `regression/tree.py:95-96`). In the heart_disease
  tree: all 2-valued attributes become `DiscreteNode`, 3- and 4-valued ones become
  `MappedDiscreteNode`.
- The `MAX_BINARIZATION = 16` limit: a discrete attribute with > 16 values and
  `binarize=True` raises `ValueError` in `fit_storage`
  (`classification/tree.py:205-212`). With `binarize=False` the same data runs.
- There is no one-hot encoding. This is a big difference from sklearn, which needs
  discrete attributes to be encoded first.

### Claim 4. Numeric threshold = a data value, not a midpoint

- Threshold candidates are the unique values present in the data; the chosen threshold
  **equals the maximum value on the left side**. Branch rule (`tree.py:119-121`): x ≤ t
  goes to branch 0, x > t to branch 1. Each side has at least `min_samples_leaf` rows.
- Evidence: 106 `NumericNode`s in the iris, heart_disease (classification) and
  heart_disease (regression) trees, all satisfying `threshold ∈ data` and
  `threshold == max(left side)`.
- sklearn reference (`criterion="entropy"`): the iris root splits the same feature with
  the same partition, but its threshold is **2.45 = (1.9 + 3.0)/2** (the midpoint),
  while Orange's is **1.9**. Predictions on training data are the same; predictions on
  new data between 1.9 and 3.0 can differ. An ExtDT that wants to be identical to
  Orange must use the "left value" convention.
- Numeric scores are multiplied by the fraction of non-NaN rows
  (`classification/tree.py:157`, `regression/tree.py:117`), see Claim 5.

### Claim 5. Missing values

- (a) Rows with a NaN target are dropped before the tree is built
  (`classification/tree.py:214`, `regression/tree.py:173`). iris with 3 NaN targets:
  `root.subset` holds 147 rows.
- (b) No imputation. When splitting, rows with NaN in the split attribute are given
  branch −1 (`classification/tree.py:119-120, 158-160`;
  `MappedDiscreteNode.branches_from_mapping` `tree.py:89-91`), so they **do not go into
  any child**; those rows are still recorded in the `subset` of the parent node.
  heart_disease, root `thal`: 303 rows, 2 NaN, the children hold 301.
- (c) Attribute scores are multiplied by the fraction of known rows (numeric
  `tree.py:157`, binarized discrete `tree.py:141`, multiway `tree.py:118`; regression
  inside the Cython scorer). Evidence: making 90 of the 150 iris `petal length` values
  NaN → its score drops to 0.4 × 0.918, and the root moves to `petal width`.
- (d) Prediction: if the value of the split attribute is NaN, traversal **stops at that
  node** and the result is the distribution of that internal node
  (`tree.py:162-170, 180-183`). iris row 0 with `petal length` NaN → probabilities
  [1/3, 1/3, 1/3] = the root distribution [50, 50, 50]. The same happens if the target
  branch is `None`.

### Claim 6. Stopping parameters

Order of checks in `_build_tree` (`classification/tree.py:183-203`,
`regression/tree.py:142-162`), for a node at level `level` (root = 1):

1. `len(node) < min_samples_leaf` → the node is not created (`None`).
2. The node becomes a leaf if any of these is true:
   `len(node) < min_samples_split`, or (classification only)
   `max(distr) >= sum(distr) * sufficient_majority`, or
   `max_depth is not None and level > max_depth`.
3. Otherwise search for a split (Claim 2); if all attributes are rejected, the node
   stays a leaf.

Evidence with the widget's default settings:

| Data | Nodes | Depth | min \|node\| | min \|internal\| | Max majority in internal |
|---|---|---|---|---|---|
| iris | 9 | 4 | 3 | 6 | 0.9074 |
| heart_disease | 77 | 11 | 2 | 5 | 0.9231 |
| heart_disease reg | 211 | 15 | 2 | 5 | (not applicable) |

- `max_depth` = the number of split levels: `max_depth` 1, 2, 3 → `depth()` 1, 2, 3
  (heart_disease).
- `sufficient_majority` does not exist in regression.
- All checkboxes off → `max_depth=None, min_samples_split=2, min_samples_leaf=1,
  sufficient_majority=1`; with `sufficient_majority=1` only pure nodes stop because of
  majority. iris: all leaves pure, depth 5.

### Claim 7. Model shape and output

- `TreeModel` (`tree.py:136`) stores `root` and `instances`. Node types
  (`tree.py:15-133`): `Node` (leaf), `DiscreteNode`, `MappedDiscreteNode`,
  `NumericNode` (has a `threshold`). Each node has `attr`, `attr_idx`, `children` (may
  contain `None`), `subset` (training row indices), `value`, and after compilation
  `parent`, `condition`, `description`.
- Classification `value` = **count per class** (not proportions), also in internal
  nodes. iris root `[50, 50, 50]`.
- Regression `value` = `[mean, var]` with var ddof=0 (`regression/tree.py:155-156`).
  heart_disease, root with target `max HR`: `[149.6073, 521.5388]`.
- Classification prediction: probability = leaf `value` / its sum, **no smoothing**
  (`tree.py:208-218`); class = argmax. Regression prediction = leaf mean. Prediction
  uses compiled code (`_compile`, `tree.py:291-361`; `compute_predictions` in Cython),
  and the result is exactly equal to the Python traversal `get_values_by_nodes`: max
  difference 0.0 on iris, heart_disease (with NaN), and regression.
- iris with default settings: 9 nodes, 5 leaves, depth 4, node types `Node` and
  `NumericNode`.

## Claims table

| No | Short claim | Main evidence |
|---|---|---|
| 1 | Widget → `Fitter` → `classification`/`regression.tree.TreeLearner`; not sklearn | `check_klaim1` |
| 2 | Info gain (bits) / MSE reduction; tie → smallest attribute index | `check_klaim2`, brute force |
| 3 | Discrete: multiway or exhaustive 2-group (≤ 16 values) | `check_klaim3` |
| 4 | Threshold = max of left side (a data value), x ≤ t goes left | `check_klaim4`, 106 nodes |
| 5 | NaN target dropped; NaN attribute does not go down to a child, score × known fraction, prediction stops | `check_klaim5` |
| 6 | Stop order: leaf → split → majority → depth | `check_klaim6` |
| 7 | `TreeModel`, value = class counts / [mean, var], prob without smoothing | `check_klaim7` |

## Status

### Handoff 26 Nov 2026 (T2b-1)

**Done**

- Read `modelling/tree.py`, `classification/tree.py`, `regression/tree.py`, `tree.py`,
  `widgets/model/owtree.py`, and `Fitter` in `modelling/base.py`.
- This document plus `docs/proofs/tree_internals_check.py`: 7 claims, all as `assert`,
  run with `D:\Orange\python.exe` → exit 0, last line `SEMUA KLAIM TERBUKTI`.
- No zestcore code yet (per the brief).

**Findings to know before designing ExtDT**

1. Orange's threshold is a data value (max of the left side), not a midpoint as in
   sklearn (Claim 4). This is a convention that has to be decided (ADR) if ExtDT wants
   to be identical to Orange.
2. Score ties are broken by attribute order (Claim 2). Ties within one attribute (two
   thresholds with the same score) have not had their tie-breaking direction proven,
   because it lives in Cython.
3. The learner's default `binarize` (`False`) differs from the widget default (`True`).
4. Preprocessors from the widget signal are dropped on the regression path (Claim 1).

**Open questions (for Raditya / Prof. Irfan)**

- ExtDT (extended decision tree) from the reference list: what is its split criterion?
  If it is not information gain, "identical to Orange" is not the right target, and
  comparing accuracy with matched settings is enough.
- The Tree widget in the GUI with an attribute of > 16 values and "Induce binary tree"
  ON: does the GUI give a warning/fallback, or an error? `owtree.py` 3.40.0 has no
  fallback code; not yet checked in the GUI.
- Tie-breaking within one attribute (see finding 2): needed if ExtDT must be identical
  1:1 on data with many repeated values.

**Next tasks (proposed, schedule set by the PM)**

- Decide the ExtDT target: identical to Orange Tree, or only comparable.
- If identical: an ADR on the threshold convention, tie-breaking, and NaN handling,
  then a `zestcore` tree starting from the iris case (purely numeric, no NaN).
