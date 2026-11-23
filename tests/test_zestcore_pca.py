"""zestcore.pca vs PCA Orange (projector widget, Normalize ON), aturan uji di
docs/orange-pca-internals.md. X untuk zestcore = hasil preprocessor Orange (ADR 0002)."""
import subprocess
import sys

import numpy as np
import pytest

from datasets import load_table
from pca_compare import (ATOL, RTOL, compare, compare_variance, orange_widget_pca,
                         preprocessed_X, zero_variance_count)
from zestcore.pca import pca

# n_null dari zero_variance_count = jumlah pasti untuk ketiganya: dijaga di
# test_pca_compare.py::test_jumlah_arah_varians_nol_sama_dengan_terukur.
DATASETS = ["iris", "heart_disease", "bank-full"]


@pytest.fixture(scope="module", params=DATASETS)
def case(request):
    table = load_table(request.param)
    proj, ref_scores = orange_widget_pca(table)
    X = preprocessed_X(table)
    # 100 = MAX_COMPONENTS widget (owpca.py:27); zestcore sendiri memotong ke min(n, p).
    return proj, ref_scores, X, zero_variance_count(table), pca(X, n_components=100)


def test_X_sama_dengan_yang_dilihat_sklearn(case):
    proj, _, X, _, got = case
    for a, b in [(proj.mean_, X.mean(axis=0)), (proj.mean_, got.mean)]:
        maks = float(np.max(np.abs(a - b)))
        assert np.allclose(a, b, rtol=RTOL, atol=ATOL), f"maks {maks:.3g}"


def test_komponen_dan_skor_identik(case):
    proj, ref_scores, _, n_null, got = case
    assert got.components.shape == proj.components_.shape
    assert got.scores.shape == ref_scores.shape
    rows = compare(proj.components_, ref_scores, got.components, got.scores, n_null)
    assert [r for r in rows if not r["allclose"]] == []


def test_explained_variance_identik(case):
    proj, _, _, n_null, got = case
    rows = compare_variance(proj.explained_variance_, got.explained_variance, n_null)
    assert [r for r in rows if not r["allclose"]] == []


def test_explained_variance_ratio_identik(case):
    proj, _, _, n_null, got = case
    rows = compare_variance(proj.explained_variance_ratio_,
                            got.explained_variance_ratio, n_null)
    assert [r for r in rows if not r["allclose"]] == []


def test_penjaga_bentuk_warning_di_luar_covariance_eigh():
    X = np.random.default_rng(0).normal(size=(50, 10))     # n < 10p
    with pytest.warns(UserWarning, match="tidak dijamin identik"):
        pca(X)


def test_penjaga_bentuk_diam_di_jalur_covariance_eigh():
    import warnings
    X = np.random.default_rng(0).normal(size=(40, 4))      # n = 10p, batas masuk
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pca(X)


def test_n_kurang_dari_2_raise():
    with pytest.raises(ValueError, match="minimal 2 baris"):
        pca(np.ones((1, 3)))


def test_n_components_dipotong_ke_min_bentuk():
    X = np.random.default_rng(0).normal(size=(3, 4))
    with pytest.warns(UserWarning, match="tidak dijamin identik"):  # penjaga bentuk: 3 < 40
        assert pca(X, n_components=100).components.shape == (3, 4)   # dipotong n
    X = np.random.default_rng(0).normal(size=(60, 4))
    assert pca(X).components.shape == (4, 4)                         # None -> min(n, p)
    assert pca(X, n_components=2).scores.shape == (60, 2)


def test_zestcore_tidak_mengimpor_orange_atau_sklearn():
    # Aturan dua lapis CLAUDE.md. Penjaga regresi: hijau sejak awal.
    code = ("import sys, zestcore.pca; "
            "bad = [m for m in ('Orange', 'sklearn') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)
