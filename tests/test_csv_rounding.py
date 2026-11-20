"""CSV Save Data vs .pkl/.npy dari run GUI yang sama.

Save Data menulis CSV dengan %g = 6 digit signifikan, jadi galat pembulatan
murni relatif: |csv - x| <= 5e-6 |x|. Karena itu rtol=1e-5 dengan atol=0;
atol > 0 akan meloloskan nilai ruang nol (~1e-15) tanpa benar-benar diuji.
"""
import numpy as np
import pytest

from evidence_io import (read_components_csv, read_components_npy,
                         read_scores_csv, read_scores_pkl)
from pca_compare import compare_csv


def test_compare_csv_pembulatan_6_digit_lolos():
    exact = np.array([1.2345649, -2.88658e-15, 0.0])
    csv = np.array([float(f"{v:g}") for v in exact])      # 1.23456, ...
    r = compare_csv(csv, exact)
    assert r["allclose"] and (r["rtol"], r["atol"]) == (1e-5, 0)
    assert r["maks_rel"] <= 5e-6


def test_compare_csv_atol_nol_menangkap_galat_di_nilai_kecil():
    # Selisih absolut 2e-15 lolos atol default 1e-8, tapi relatifnya 70%.
    r = compare_csv(np.array([-0.88658e-15]), np.array([-2.88658e-15]))
    assert not r["allclose"]


PREFIXES = ["iris_pca_norm-on_4pc", "heart_pca_norm-on_25pc"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_scores_csv_vs_pkl(prefix):
    csv = read_scores_csv(f"{prefix}_scores.csv")
    exact = read_scores_pkl(f"{prefix}_scores.pkl")
    assert csv.shape == exact.shape
    assert compare_csv(csv, exact)["allclose"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_components_csv_vs_npy(prefix):
    csv = read_components_csv(f"{prefix}_components.csv")
    exact = read_components_npy(f"{prefix}_components.npy")
    assert csv.shape == exact.shape
    assert compare_csv(csv, exact)["allclose"]
