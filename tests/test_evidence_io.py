"""Loader bukti GUI Orange 3.40.0 di evidence/pca/.

Nilai harapan adalah literal yang dibaca langsung dari file (baris pertama CSV),
bukan dihitung ulang dengan cara yang sama seperti loader.
"""
import numpy as np

from evidence_io import (read_components_csv, read_components_npy,
                         read_scores_csv, read_scores_pkl)

IRIS_SCORES_ROW0 = [-2.26454, 0.505704, 0.121943, -0.0230733]
IRIS_PC1 = [0.522372, -0.263355, 0.581254, 0.565611]  # tanpa variance 0.727705


def test_scores_csv_hanya_kolom_pc_tanpa_kelas():
    X = read_scores_csv("iris_pca_norm-on_4pc_scores.csv")
    assert X.shape == (150, 4)
    assert X.dtype == np.float64
    assert X[0].tolist() == IRIS_SCORES_ROW0


def test_components_csv_tanpa_kolom_nama_dan_variance():
    C = read_components_csv("iris_pca_norm-on_4pc_components.csv")
    assert C.shape == (4, 4)
    assert C[0].tolist() == IRIS_PC1


# Literal CSV = nilai yang sama ditulis %g (6 digit signifikan): galat relatif
# <= 5e-6, jadi rtol=1e-5 dengan atol=0.
def test_scores_pkl_lewat_orange_table():
    X = read_scores_pkl("iris_pca_norm-on_4pc_scores.pkl")
    assert X.shape == (150, 4)
    assert np.allclose(X[0], IRIS_SCORES_ROW0, rtol=1e-5, atol=0)


def test_components_npy_lewat_np_load():
    C = read_components_npy("iris_pca_norm-on_4pc_components.npy")
    assert C.shape == (4, 4)
    assert np.allclose(C[0], IRIS_PC1, rtol=1e-5, atol=0)
