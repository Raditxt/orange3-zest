"""zestcore.transform vs jalur compute_value Orange untuk data baru (ADR 0003).

Seam A: projector widget PCA di-fit pada baris genap (latih); baris ganjil jadi data
baru. Referensi = model(test).X, jalur yang sama dengan Apply Domain
(Orange/projection/base.py:118-119, 139-142). Input zestcore = data baru yang diproses
dengan statistik latih (test.transform(model.pre_domain)), plus mean/components dari
model.proj. Karena mean/components sama di kedua sisi, yang diuji hanya rumus
transform: np.allclose default pada seluruh matriks skor, tanpa aturan seri/ruang nol.

Catatan split: heart_disease setengah punya n_latih < 10p, jadi Orange memakai solver
'full', bukan covariance_eigh (ADR 0001). Tidak masalah di sini karena mean/components
diambil dari model.proj. Test widget dengan data baru (langkah 4) harus memakai split
dengan n_latih >= 10p, atau iris saja.
"""
import numpy as np
import pytest

from datasets import load_table
from pca_compare import ATOL, RTOL, _widget_projector
from zestcore.pca import transform

DATASETS = ["iris", "heart_disease", "bank-full"]


@pytest.mark.parametrize("name", DATASETS)
def test_transform_data_baru_sama_dengan_apply_domain(name):
    table = load_table(name)
    train, test = table[::2], table[1::2]
    model = _widget_projector()(train)
    ref = model(test).X
    X_new = test.transform(model.pre_domain).X

    got = transform(X_new, model.proj.mean_, model.proj.components_)

    assert got.shape == ref.shape
    maks = float(np.max(np.abs(got - ref)))
    assert np.allclose(got, ref, rtol=RTOL, atol=ATOL), f"maks {maks:.3g}"


@pytest.mark.parametrize("name", ["iris", "heart_disease"])
def test_transform_satu_baris_data_baru(name):
    """Seam B: satu baris data baru, bentuk (1, p). pca() menolak n < 2, transform tidak."""
    table = load_table(name)
    train, test = table[::2], table[1::2]
    model = _widget_projector()(train)
    ref = model(test[:1]).X
    X_new = test[:1].transform(model.pre_domain).X

    got = transform(X_new, model.proj.mean_, model.proj.components_)

    assert got.shape == ref.shape == (1, len(model.proj.components_))
    maks = float(np.max(np.abs(got - ref)))
    assert np.allclose(got, ref, rtol=RTOL, atol=ATOL), f"maks {maks:.3g}"
