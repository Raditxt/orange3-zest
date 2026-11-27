# GUI evidence: Orange 3.40.0 PCA and Zest PCA

Evidence for the Zest PCA widget (T2a) is in the section
[GUI evidence for T2a](#gui-evidence-for-t2a-zest-pca-vs-orange-pca-heart_disease-27-nov-2026).
The rest of this file covers Stage 1 evidence: PCA widget output recorded from the Orange
GUI, used to show that Orange PCA run from a script (projector widget) matches what the
GUI produces. The comparisons are in `notebooks/01_pca_gui_evidence.ipynb` (iris,
heart_disease), `notebooks/02_pca_perbandingan.ipynb` (all three datasets) and `tests/`
(test rules: `docs/orange-pca-internals.md`).

## Recording environment

| | Version |
|---|---|
| Orange | 3.40.0 (GUI and Orange's bundled Python) |
| Python | 3.12.12 |
| NumPy | 2.1.3 |
| scikit-learn | 1.5.2 |
| OS | Windows 11 |

Extra packages in Orange's Python, only for tests and notebooks (not used by the GUI):

| Package | Version |
|---|---|
| pytest | 9.2.0 |
| nbclient | 0.11.0 |
| nbformat | 5.11.1 |

## Workflow and widget settings

All three workflows have the same shape (`*_workflow.png`): File → PCA → Save Data
(Transformed Data), Save Data (1) (Components), Python Script (Components). Settings as
read from the `.ows`:

| Widget | Setting | iris | heart_disease | bank-full |
|---|---|---|---|---|
| File | dataset | `Orange/datasets/iris.tab` (built-in) | `Orange/datasets/heart_disease.tab` (built-in) | `bank-full.csv` (UCI, outside the repo), `y` = target |
| PCA | Normalize variables | ON | ON | ON |
| PCA | Components | 4 | 25 | 51 |
| PCA | Explained variance | 100% | 100% | 100% |
| PCA | Apply Automatically | ON | ON | ON |
| Save Data, Save Data (1) | Add type annotations | OFF | OFF | OFF |
| Save Data, Save Data (1) | Autosave | OFF | OFF | OFF |

Components = the maximum number of components for the data (min(n, p)), so no component
is cut off. The PCA settings are also visible in `*_settings.png`.

## File list

File names: `<data>_pca_norm-on_<k>pc_<output>.<ext>`.

| File | PCA output | Written by | Contents |
|---|---|---|---|
| `*_scores.pkl` | Transformed Data | Save Data | Orange `Table`: scores PC1..PCk + class (full precision) |
| `*_scores.csv` | Transformed Data | Save Data | class, PC1..PCk; no type annotation rows; numbers in `%g` (6 significant digits) |
| `*_scores.csv.metadata` | Transformed Data | Save Data | table attributes (name, description, dataset reference), written automatically next to the CSV |
| `*_components.npy` | Components | Python Script | `components_` k × p, float64, full precision |
| `*_components.csv` | Components | Save Data (1) | `components`, `variance` (variance ratio), then loadings per feature; `%g` |
| `*_pca.ows` | – | Orange | full workflow with widget settings |
| `*_settings.png` | – | screenshot | PCA widget window |
| `*_workflow.png` | – | screenshot | workflow canvas |

`<data>` = `iris` (150 × 4, k = 4), `heart` (heart_disease 303 × 25 after Continuize,
k = 25) and `bank` (bank-full 45211 × 51 after Continuize, k = 51). For bank there is
only `*_scores.pkl`; the score CSV (45211 rows) was not created.

## GUI evidence for T2a: Zest PCA vs Orange PCA (heart_disease, 27 Nov 2026)

The heart_disease case workflow (2 PCs → Logistic Regression), built once with Orange
PCA and once with Zest PCA. Settings, results, and the answer are in
`docs/orange-pca-internals.md`, section "GUI evidence for T2a".

| File | Contents |
|---|---|
| `heart_pca_orange.ows` | Orange branch: File → PCA (25 PCs, Normalize ON) → Select Columns (PC1, PC2) → Test and Score (Logistic Regression, 10-fold CV); Scatter Plot, Data Table Components; Test and Score for the original columns |
| `heart_pca_zest.ows` | Zest branch, same as above, plus Save Data (scores `.pkl`), Python Script (components `.npy`), and Data Sampler (100 rows) → Zest PCA (1) for the warning |
| `heart_zestpca_norm-on_25pc_scores.pkl` | Zest PCA Transformed Data, full-precision `Table` (303 × 25 + class) |
| `heart_zestpca_norm-on_25pc_components.npy` | Zest PCA Components, `components_` 25 × 25 float64 |
| `compare_zest_gui.py` | compares the two files above with `heart_pca_norm-on_25pc_*` (Orange PCA reference) using `compare()` |
| `heart_pca_orange_workflow.png`, `heart_pca_zest_workflow.png` | canvas of both workflows |
| `heart_pca_orange_scatter.png`, `heart_pca_zest_scatter.png` | Scatter Plot PC1 × PC2, colored by `diameter narrowing` |
| `heart_pca_zest_testscore.png` | Test and Score of the Zest branch: 2 PCs (AUC 0.900, CA 0.818) and original columns (AUC 0.909, CA 0.845) |
| `heart_zestpca_warning.png` | Zest PCA (1) window on 100 rows (100 × 25, n < 10p): ADR 0001 warning text in the bottom status bar, output 100 \| 2; warning icon on the canvas node |
| `zest_category.png` | Zest category in Orange's widget panel |

The full-precision evidence is the `.pkl` and `.npy`. No CSV was created for T2a. Run
from the repo root:

```powershell
D:\Orange\python.exe evidence\pca\compare_zest_gui.py
# [norm-on_25pc] IDENTIK: komponen (25, 25), skor (303, 25), selisih maks 8.88e-16
```

In `heart_pca_zest_testscore.png` and `heart_pca_zest_scatter.png` the top-right corner
is covered with a white box. In `heart_zestpca_warning.png` white boxes cover the areas
above, to the right of, and below the Zest PCA (1) window (the top-right and bottom-right
corners of the image). Those areas contained windows of other applications that were
captured by accident and are not part of Orange. The contents of the Orange windows are
unchanged.

The `np.save` path in the Python Script of `heart_pca_zest.ows` is absolute (the repo
folder on the recording machine). Adjust it if the workflow is opened on another
machine.

## Notes

- **bank-full is not stored in the repo** (see `docs/orange-pca-internals.md`, scope
  limits). The File widget reads `bank-full.csv` with `y` as the target, so the 16
  attributes become 51 columns. Tests and notebooks read it through `tests/datasets.py`
  (`../magang-datamining/datasets/bank-full.csv` or the `ZEST_BANK_FULL` environment
  variable).
- **Save Data fails to write `.pkl` for the Components output.** The resulting file is
  0 bytes. Full-precision components were therefore taken through the Python Script
  widget:

  ```python
  import numpy as np
  np.save(r"<repo>\evidence\pca\<data>_pca_norm-on_<k>pc_components.npy", in_data.X)
  print(in_data.X.shape, in_data.X.dtype)
  out_data = in_data
  ```

  The 0-byte file is not stored in the repo. This script is saved in the `.ows` as the
  `scriptText` of the Python Script node, not in the widget's script library (the
  library only contains the "Table from numpy" template). Open the `.ows` in Orange to
  see it.
- **`.ows` records only one file name per Save Data.** The Save Data (scores) node keeps
  only the last `stored_name` (iris: `.pkl`, heart: `.csv`), even though both score
  formats exist for both datasets.
- **The GUI does not output `explained_variance_`**, only its ratio
  (`explained_variance_ratio_`, the `variance` column in `*_components.csv`). Variance
  from the GUI can only be tested through that ratio; `explained_variance_` itself is
  tested in scripts.
- **The script reference uses Orange's private API.** `OWPCA._create_projector` (Orange
  3.40.0) is used to create exactly the same projector as the widget; it may change in
  other Orange versions.
- **CSV = `%g` rounding.** Compared with `.pkl`/`.npy`, the maximum relative difference
  is ≤ 5e-6 (rtol=1e-5, atol=0). For full-precision comparison use `.pkl`/`.npy`.
- **GUI vs script.** On the recording machine, `.pkl`/`.npy` are exactly equal to Orange
  PCA run from a script (maximum difference 0.0). That is an observation, not a
  requirement; the test requirement is `np.allclose` (rtol=1e-5, atol=1e-8), because a
  different machine (different BLAS) may show differences of ~1e-16.

## Running tests and notebooks

From the repo root, in PowerShell, with Orange's bundled Python:

```powershell
# once: test and notebook packages
D:\Orange\python.exe -m pip install pytest nbclient nbformat

# tests
D:\Orange\python.exe -m pytest tests -q

# re-execute a notebook (output saved in the same file)
$env:PYTHONUTF8 = "1"
D:\Orange\Scripts\jupyter-execute.exe --inplace notebooks\01_pca_gui_evidence.ipynb
```

CI (`.github/workflows/tests.yml`, GitHub Actions, ubuntu-latest) runs the same tests
with the package versions above and Qt `offscreen`. bank-full is not available on the
runner, so the bank tests are skipped in CI; they only run on a local machine that has
the data.

Replace `D:\Orange` with your Orange installation folder. `PYTHONUTF8=1` is required on
Windows: without it `jupyter-execute` reads the notebook as cp1252 and corrupts non-ASCII
characters (`≤` becomes `â‰¤`). `python -m jupyter execute` does not run on this
installation (exits with code 1 and no message); use `jupyter-execute.exe`.
