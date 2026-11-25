"""Adaptor Orange ZestPCA/ZestPCAModel vs projector widget PCA (ADR 0003, opsi 1c).

Skor kedua sisi diambil lewat jalur compute_value yang sama dengan OWPCA
(model(data), owpca.py:321). Komponen Zest berasal dari zestcore, jadi dibandingkan
dengan compare() lengkap (aturan seri + ruang nol, docs/orange-pca-internals.md).
n_null = zero_variance_count: sama dengan jumlah terukur untuk data penuh
(test_pca_compare.py) dan subset latih di bawah (dicek 25 Nov 2026).
"""
import pickle

import numpy as np
import pytest
from Orange.preprocess import Normalize

from datasets import load_table
from pca_compare import _widget_projector, compare, compare_variance, zero_variance_count
from orangecontrib.zest.pca import ZestPCA


def zest_projector():
    """ZestPCA dengan setting default projector widget PCA, Normalize ON
    (owpca.py:278-282)."""
    return ZestPCA(n_components=100,
                   preprocessors=ZestPCA.preprocessors + [Normalize(center=False)])


def failed_rows(ref, got, ref_scores, got_scores, n_null):
    """Baris compare/compare_variance yang tidak lolos toleransi."""
    rows = compare(ref.components_, ref_scores, got.components_, got_scores, n_null)
    rows += compare_variance(ref.explained_variance_, got.explained_variance_, n_null)
    rows += compare_variance(ref.explained_variance_ratio_,
                             got.explained_variance_ratio_, n_null)
    return [r for r in rows if not r["allclose"]]


@pytest.mark.parametrize("name", ["iris", "heart_disease", "bank-full"])
def test_data_penuh_sama_dengan_owpca(name):
    table = load_table(name)
    ref, got = _widget_projector()(table), zest_projector()(table)
    ref_t, got_t = ref(table), got(table)

    assert [a.name for a in got_t.domain.attributes] == \
        [a.name for a in ref_t.domain.attributes]
    assert failed_rows(ref, got, ref_t.X, got_t.X, zero_variance_count(table)) == []


@pytest.mark.parametrize("name", ["iris", "heart_disease"])
def test_normalize_off_sama_dengan_owpca(name):
    """Preprocessor bawaan ZestPCA = projector widget dengan Normalize tidak dicentang."""
    table = load_table(name)
    ref = _widget_projector(normalize=False)(table)
    got = ZestPCA(n_components=100)(table)
    ref_t, got_t = ref(table), got(table)

    assert failed_rows(ref, got, ref_t.X, got_t.X, zero_variance_count(table)) == []


# n_latih >= 10p supaya Orange tetap di jalur covariance_eigh (ADR 0001):
# iris genap 75 >= 40; heart_disease 250 = 10 * 25.
SPLITS = {"iris": (slice(None, None, 2), slice(1, None, 2)),
          "heart_disease": (slice(None, 250), slice(250, None))}


def split(name):
    table = load_table(name)
    train_rows, test_rows = SPLITS[name]
    return table[train_rows], table[test_rows]


@pytest.mark.parametrize("name", SPLITS)
def test_data_baru_lewat_apply_domain(name):
    """Apply Domain = data.transform(template.domain) (owtransform.py:27). Preprocessing
    data baru memakai statistik latih yang tersimpan di pre_domain."""
    train, test = split(name)
    ref, got = _widget_projector()(train), zest_projector()(train)

    ref_new = test.transform(ref(train).domain).X
    got_new = test.transform(got(train).domain).X

    assert got_new.shape == ref_new.shape == (len(test), len(ref.components_))
    assert failed_rows(ref, got, ref_new, got_new, zero_variance_count(train)) == []


def test_pickle_bolak_balik():
    """Domain hasil (isi .pkl Save Data) dan model tetap bisa memproyeksikan data baru
    setelah pickle. Proses dan jalur hitung sama, jadi hasilnya harus bit-identik."""
    train, test = split("iris")
    model = zest_projector()(train)
    template = model(train)
    before = test.transform(template.domain).X

    template_back = pickle.loads(pickle.dumps(template))
    model_back = pickle.loads(pickle.dumps(model))

    assert not np.isnan(before).any()
    assert np.array_equal(test.transform(template_back.domain).X, before)
    assert np.array_equal(model_back(test).X, before)
