"""PCA NumPy murni: langkah 4-7 docs/orange-pca-internals.md (jalur covariance_eigh
sklearn 1.5.2, ADR 0001). X harus sudah diproses pemanggil (ADR 0002)."""
import warnings
from typing import NamedTuple

import numpy as np


class PCAResult(NamedTuple):
    mean: np.ndarray                      # (p,)
    components: np.ndarray                # (k, p), baris = komponen
    explained_variance: np.ndarray        # (k,)
    explained_variance_ratio: np.ndarray  # (k,)
    scores: np.ndarray                    # (n, k)


def pca(X, n_components=None):
    """PCA X (n x p, numerik, tanpa NaN) seperti sklearn PCA(svd_solver='auto')
    di Orange 3.40.0 untuk bentuk p <= 1000 dan n >= 10p.

    k = min(n, p), atau min(n_components, n, p) bila diisi. Widget Orange memakai
    n_components=100 (owpca.py:27), jadi untuk meniru widget pada p > 100 pemanggil
    harus mengirim n_components=100.

    Kovarians dihitung persis seperti sklearn (_pca.py:611-617): C = X.T @ X, lalu
    C -= (n * mean_kolom) * mean_baris, lalu C /= n - 1. Bentuk ini setara matematis
    dengan Xc.T @ Xc / (n - 1) (Xc = X - mean), tetapi hasil floating-point-nya tidak
    sama. Urutan perkalian (n·a)·b juga diikuti: n * outer(mean, mean) = n·(a·b)
    mengubah digit terakhir C. Bentuk sklearn yang dipakai supaya matriks C sama
    dengan milik Orange, karena targetnya identik, bukan sekadar setara.

    Skor dihitung X @ components.T - mean @ components.T, juga bentuk sklearn
    (decomposition/_base.py:149, 156), yang dipanggil Orange lewat transform
    (Orange/projection/base.py:115-116).

    Tanda: tiap baris components dibalik supaya elemen |terbesar| positif (indeks
    pertama bila seri), sama dengan svd_flip(u_based_decision=False).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n < 2: raise ValueError(f"PCA butuh minimal 2 baris, dapat n = {n}")
    k = min(n, p) if n_components is None else min(n_components, n, p)
    if not (p <= 1000 and n >= 10 * p):
        warnings.warn(
            f"X {n}x{p} tidak memenuhi p <= 1000 dan n >= 10p: Orange memakai jalur "
            "'full' atau 'randomized' untuk bentuk ini, jadi hasil zestcore hanya "
            "mendekati dan tidak dijamin identik (ADR 0001).", stacklevel=2)

    mean = X.mean(axis=0)
    C = X.T @ X
    C -= n * mean.reshape(-1, 1) * mean.reshape(1, -1)   # (n·a)·b, urutan _pca.py:613-616
    C /= n - 1
    eigenvals, eigenvecs = np.linalg.eigh(C)
    eigenvals = eigenvals[::-1].copy()       # menurun
    components = eigenvecs[:, ::-1].T.copy()
    eigenvals[eigenvals < 0.0] = 0.0         # galat numerik, C semidefinit positif (_pca.py:637)

    # svd_flip berbasis baris (_pca.py:647, extmath.py:898-905)
    max_abs = np.argmax(np.abs(components), axis=1)
    components *= np.sign(components[np.arange(p), max_abs])[:, None]

    ratio = eigenvals / eigenvals.sum()      # dibagi semua eigenvalue, sebelum diiris
    components = components[:k]
    scores = X @ components.T
    scores -= mean @ components.T
    return PCAResult(mean, components, eigenvals[:k], ratio[:k], scores)
