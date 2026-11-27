# Orange3-Zest

An Orange Data Mining add-on that lets non-expert users run research models (especially ones Orange does not have yet, such as metaheuristics) through widgets. Internship project of Raditya, supervised by Prof. Irfan. Public repo, licensed under GPL-3.0.

## Current stage

Stage 1 is a proof of concept: a pure-NumPy PCA whose results match Orange's built-in PCA 1:1, demonstrated in notebooks. The Zest PCA widget (task T2a, Prof. Irfan's direction of 23 Nov 2026) wraps `zestcore.pca` and serves as the add-on template for the next model (ExtDT); preprocessing goes through Orange preprocessors (docs/adr/0002-batas-preprocessing.md). Stage 2 is the first model from Prof. Irfan's reference list (Firefly, Komodo Mlipir, Al-Biruni Earth Radius, PSO, extended decision tree), delivered as a widget.

## Two-layer structure (hard rule)

- `zestcore/`: algorithm core, pure NumPy. `import Orange` and `import sklearn` are FORBIDDEN here.
- `orangecontrib/zest/`: Orange widget layer, only wraps `zestcore`. `orangecontrib/` is a namespace package; do not add an `__init__.py` directly inside it.
- `tests/`: pytest. scikit-learn and Orange may be used here as references.
- `notebooks/`, `docs/`, `evidence/` (CSV + .ows from the Orange GUI).
- PCA specification: docs/orange-pca-internals.md; design decisions: docs/adr/.

## Environment

All work runs on Orange's bundled Python so the numbers match the GUI:

- Interpreter: `D:\Orange\python.exe`
- Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2, NumPy 2.1.3
- Installed Orange source: `D:\Orange\Lib\site-packages\Orange\`
- Shell: PowerShell (Windows)

Refer to the source of the installed version above, not to memory or the GitHub master branch.

## Technical principles

- "Identical" must be proven with numbers: `np.allclose` plus the reported maximum difference. Always state what is compared against, the tolerance, and the settings.
- PCA component signs can flip between implementations. Use a deterministic sign convention and document it.
- If results turn out not to be identical, that is a valid finding. Explain the cause; do not hide it or cover it with a loose tolerance.
- Simple first: small changes, every step checkable, no abstractions before they are needed.

## Language

- Repo documentation (all `.md` files, including this one) and commit messages: English.
- Conversation with Raditya: Indonesian, plain and direct.
- Text shown in the Orange GUI (widget description, Warning/Error messages, `ShapeWarning` text) stays Indonesian for now; changing it invalidates the GUI screenshots in `evidence/`.
- `LICENSE` is the official GPL-3.0 text from the FSF, unmodified and untranslated.

## Working with Raditya

- Raditya aims for a 50:50 split of work with AI, and must understand and be able to defend every line of code in front of the supervisor.
- Before writing code, briefly explain the approach. For new material (SVD, PCA, Orange widget architecture), give a small example that can be run directly.
- Work in small steps. After each step, explain what changed and why, then wait for Raditya before continuing.
- The internship report (chapters I–V) is written by Raditya. Helping with the outline and checking technical accuracy is fine, but do not write the report in full.
