"""Bukti GUI (.pkl/.npy) vs PCA Orange lewat skrip, aturan uji di
docs/orange-pca-internals.md. Angka harapan diambil dari spesifikasi (Klaim 5, 7)."""
import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table

import pytest

from pca_compare import compare, compare_ratio, orange_widget_pca, tied_rows, zero_variance_count


def test_jumlah_arah_varians_nol_dari_metadata():
    # Klaim 7: heart_disease 7 diskrit, semua nilai deklarasi muncul -> 7.
    assert zero_variance_count(Table("heart_disease")) == 7
    assert zero_variance_count(Table("iris")) == 0


@pytest.mark.parametrize("dataset", ["iris", "heart_disease"])
def test_jumlah_arah_varians_nol_sama_dengan_terukur(dataset):
    # Rumus hanya batas bawah (aturan uji poin 2); compare() memakainya sebagai
    # jumlah pasti, jadi untuk data yang dipakai harus sama dengan yang terukur.
    table = Table(dataset)
    proj, _ = orange_widget_pca(table)
    assert zero_variance_count(table) == int((proj.explained_variance_ < 1e-10).sum())


def test_jumlah_arah_varians_nol_semua_cabang_rumus():
    # Data sintetis Klaim 7 (docs/proofs/pca_internals_check.py): harapan 6,
    # sama dengan jumlah explained_variance_ < 1e-10 yang terukur di sana.
    rng = np.random.default_rng(0)
    n = 50
    kolom = [
        (ContinuousVariable("k1"), rng.normal(size=n)),
        (ContinuousVariable("k2"), rng.normal(size=n)),
        (ContinuousVariable("konst"), np.full(n, 7.0)),                    # 1
        (DiscreteVariable("A", values=("a", "b", "c")),
         np.r_[rng.integers(0, 3, n - 3), [np.nan] * 3]),                  # 1
        (DiscreteVariable("B", values=("a", "b", "c")),
         rng.integers(0, 2, n).astype(float)),                             # 2
        (DiscreteVariable("C", values=("a", "b")), np.zeros(n)),           # 2
        (DiscreteVariable("D", values=("a", "b")), np.full(n, np.nan)),    # 0
        (DiscreteVariable("E", values=("a",)), np.zeros(n)),               # 0
    ]
    sint = Table.from_numpy(Domain([v for v, _ in kolom]),
                            np.column_stack([x for _, x in kolom]))
    assert zero_variance_count(sint) == 6


def test_komponen_seri_hanya_dari_variabel_biner():
    # Klaim 5: di luar ruang nol, heart_disease PC2 (gender) dan PC3 (fasting
    # blood sugar) seri; iris tidak ada. Baris ruang nol (PC19, 22, 24, 25) juga
    # seri, tapi tidak dibandingkan (aturan uji poin 2), jadi tidak diuji di sini.
    heart = Table("heart_disease")
    proj, scores = orange_widget_pca(heart)
    comp = proj.components_
    assert comp.shape == (25, 25) and scores.shape == (303, 25)
    n_nonnull = len(comp) - zero_variance_count(heart)
    assert tied_rows(comp[:n_nonnull]) == [1, 2]
    assert tied_rows(orange_widget_pca(Table("iris"))[0].components_) == []


@pytest.fixture(scope="module")
def heart_ref():
    heart = Table("heart_disease")
    proj, scores = orange_widget_pca(heart)
    return proj.components_, scores, zero_variance_count(heart)


def _find(rows, jenis, pc):
    [r] = [r for r in rows if r["jenis"] == jenis and r["pc"] == pc]
    return r


def test_compare_identik_semua_lolos_dan_selisih_dicatat(heart_ref):
    comp, scores, n_null = heart_ref
    rows = compare(comp, scores, comp.copy(), scores.copy(), n_null)
    assert all(r["allclose"] for r in rows)
    assert all(r["maks"] == 0.0 for r in rows)
    # 18 komponen + 25 skor; komponen ruang nol (PC19-25) tidak dibandingkan.
    assert len(rows) == 18 + 25
    assert _find(rows, "skor", 19)["rtol"] == 0 and _find(rows, "skor", 19)["atol"] == 1e-8


def test_compare_tanda_terbalik_di_komponen_tidak_seri_gagal(heart_ref):
    comp, scores, n_null = heart_ref
    c, x = comp.copy(), scores.copy()
    c[0], x[:, 0] = -c[0], -x[:, 0]                       # PC1 tidak seri
    rows = compare(comp, scores, c, x, n_null)
    assert not _find(rows, "komponen", 1)["allclose"]
    assert not _find(rows, "skor", 1)["allclose"]


def test_compare_tanda_terbalik_di_komponen_seri_lolos_dengan_s(heart_ref):
    comp, scores, n_null = heart_ref
    c, x = comp.copy(), scores.copy()
    c[1], x[:, 1] = -c[1], -x[:, 1]                       # PC2 seri
    rows = compare(comp, scores, c, x, n_null)
    assert all(r["allclose"] for r in rows)
    assert _find(rows, "komponen", 2)["s"] == -1 and _find(rows, "skor", 2)["s"] == -1


def test_compare_s_dari_komponen_juga_dipakai_untuk_skor(heart_ref):
    # Komponen dibalik tapi skor tidak: s = -1 membuat skor gagal (bukan "or").
    comp, scores, n_null = heart_ref
    c = comp.copy()
    c[1] = -c[1]
    rows = compare(comp, scores, c, scores.copy(), n_null)
    assert _find(rows, "komponen", 2)["allclose"]
    assert not _find(rows, "skor", 2)["allclose"]


def test_compare_ruang_nol_hanya_skor_dengan_atol(heart_ref):
    comp, scores, n_null = heart_ref
    c, x = comp.copy(), scores.copy()
    c[18:] = 0.5                                           # vektor nol sembarang
    x[:, 18:] += 5e-9                                      # < atol 1e-8
    assert all(r["allclose"] for r in compare(comp, scores, c, x, n_null))
    x[:, 18:] += 1e-8                                      # total 1.5e-8 > atol
    rows = compare(comp, scores, c, x, n_null)
    assert not any(_find(rows, "skor", k)["allclose"] for k in range(19, 26))


def test_compare_ratio_non_nol_relatif_ruang_nol_absolut():
    # Non-nol: pembulatan %g, rtol=1e-5 atol=0. Ruang nol: rasio ~1e-16 adalah noise
    # numerik, jadi aturan yang sama dengan skor ruang nol (rtol=0, atol=1e-8).
    ref = np.array([0.6, 0.4, 1.5e-16, 0.0])
    rows = compare_ratio(np.array([0.600001, 0.4, 3e-16, 0.0]), ref, n_null=2)
    assert [(r["pc"], r["rtol"], r["atol"], r["allclose"]) for r in rows] == [
        ("PC1-2", 1e-5, 0, True), ("PC3-4", 0, 1e-8, True)]
    rows = compare_ratio(np.array([0.60001, 0.4, 0.0, 2e-8]), ref, n_null=2)
    assert [r["allclose"] for r in rows] == [False, False]


def test_compare_komponen_seri_tegak_lurus_raise(heart_ref):
    # a_k . b_k = 0 -> s = sign(0) = 0: tanda tidak bisa ditentukan, jangan
    # diam-diam membandingkan dengan vektor nol.
    comp, scores, n_null = heart_ref
    c = comp.copy()
    c[1] = 0.0                                             # a . b tepat 0
    with pytest.raises(ValueError, match="PC2"):
        compare(comp, scores, c, scores.copy(), n_null)
