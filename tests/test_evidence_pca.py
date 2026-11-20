"""Bukti GUI Orange 3.40.0 (evidence/pca/) vs PCA Orange lewat skrip.

Acuan: projector widget PCA (Normalize ON) dijalankan pada dataset bawaan yang
sama dengan File widget di .ows (iris.tab, heart_disease.tab).
"""
import pytest
from Orange.data import Table

from evidence_io import read_components_csv, read_components_npy, read_scores_pkl
from pca_compare import compare, compare_ratio, orange_widget_pca, zero_variance_count

CASES = [("iris", "iris_pca_norm-on_4pc"),
         ("heart_disease", "heart_pca_norm-on_25pc")]


@pytest.mark.parametrize("dataset, prefix", CASES)
def test_gui_pkl_npy_identik_dengan_skrip(dataset, prefix):
    table = Table(dataset)
    ref_comp, ref_scores, _ = orange_widget_pca(table)
    gui_comp = read_components_npy(f"{prefix}_components.npy")
    gui_scores = read_scores_pkl(f"{prefix}_scores.pkl")
    assert gui_comp.shape == ref_comp.shape and gui_scores.shape == ref_scores.shape
    rows = compare(ref_comp, ref_scores, gui_comp, gui_scores,
                   zero_variance_count(table))
    assert [r for r in rows if not r["allclose"]] == []


@pytest.mark.parametrize("dataset, prefix", CASES)
def test_gui_rasio_varians_csv_vs_skrip(dataset, prefix):
    # GUI tidak mengeluarkan explained_variance_, hanya rasionya (kolom `variance`
    # di CSV Components). Aturan uji 1 jadi hanya bisa diuji lewat rasio.
    table = Table(dataset)
    _, _, ref_ratio = orange_widget_pca(table)
    _, gui_ratio = read_components_csv(f"{prefix}_components.csv")
    rows = compare_ratio(gui_ratio, ref_ratio, zero_variance_count(table))
    assert [r for r in rows if not r["allclose"]] == []
