# Orange3-Zest

[![tests](https://github.com/Raditxt/orange3-zest/actions/workflows/tests.yml/badge.svg)](https://github.com/Raditxt/orange3-zest/actions/workflows/tests.yml)

Add-on [Orange Data Mining](https://orangedatamining.com/) yang membawa model-model dari
penelitian, terutama yang belum ada di Orange (misalnya algoritma metaheuristik), ke dalam
widget supaya bisa dipakai tanpa menulis kode.

Proyek magang, dikerjakan bersama asisten AI (Claude Code). Instruksi kerja untuk AI ada
di [`CLAUDE.md`](CLAUDE.md).

## Status

| Tahap | Isi | Status |
|---|---|---|
| 1 | Proof of concept: PCA NumPy murni yang hasilnya identik dengan PCA bawaan Orange | Selesai |
| 2 | Model pertama dari daftar rujukan (kandidat: Firefly, Komodo Mlipir, Al-Biruni Earth Radius, PSO, extended decision tree), dalam bentuk widget | Kajian kandidat |

Tahap 1 sengaja tidak membuat widget PCA, karena Orange sudah punya. Tujuannya membuktikan
bahwa algoritma bisa ditulis ulang di luar Orange dan hasilnya dapat dibuktikan sama,
sebelum masuk ke model yang belum ada di Orange. Karena itu add-on ini **belum
menyediakan widget**.

## Hasil Tahap 1

`zestcore.pca.pca` dibandingkan dengan PCA Orange 3.40.0 (widget PCA, Normalize ON) pada
tiga dataset. Angka di bawah adalah selisih absolut maksimum terhadap Orange:

| Dataset | Pembanding | komponen | skor | `explained_variance_` |
|---|---|---|---|---|
| iris (150×4) | zestcore | 0.0 | 5.55e-17 | 0.0 |
| heart_disease (303×25) | zestcore | 0.0 | 8.88e-16 | 0.0 |
| bank-full (45211×51) | zestcore | 0.0 | 3.11e-15 | 0.0 |

Komponen dan varians sama persis (0.0). Skor berbeda di orde 1e-15, yaitu galat
pembulatan floating-point dari urutan operasi BLAS, jauh di bawah toleransi uji
(`np.allclose`, rtol=1e-5, atol=1e-8). Pembanding lain (scikit-learn langsung dan output
GUI Orange yang direkam), perlakuan tanda komponen, dan komponen di ruang nol dibahas di
[`docs/orange-pca-internals.md`](docs/orange-pca-internals.md) dan
[`notebooks/02_pca_perbandingan.ipynb`](notebooks/02_pca_perbandingan.ipynb).

## Struktur repo

Dua lapis, dengan batas yang dijaga ketat:

| Folder | Isi |
|---|---|
| `zestcore/` | Inti algoritma, **NumPy murni**. Tidak boleh `import Orange` atau `import sklearn`, supaya algoritmanya bisa diperiksa dan diuji tanpa Orange. |
| `orangecontrib/zest/` | Lapis widget Orange yang hanya membungkus `zestcore` (masih kosong sampai Tahap 2). |
| `tests/` | pytest. Orange dan scikit-learn dipakai di sini sebagai pembanding. |
| `notebooks/` | Notebook bukti: `01_pca_gui_evidence` (GUI vs skrip), `02_pca_perbandingan` (zestcore, scikit-learn, GUI vs Orange). |
| `evidence/pca/` | Output widget PCA yang direkam dari GUI Orange (`.pkl`, `.npy`, CSV, `.ows`, screenshot). Lihat [README-nya](evidence/pca/README.md). |
| `docs/` | Spesifikasi PCA Orange dan keputusan desain (ADR). |

### Dokumentasi

- [`docs/orange-pca-internals.md`](docs/orange-pca-internals.md): spesifikasi cara Orange
  3.40.0 menjalankan PCA, langkah yang ditiru, aturan test, batas cakupan, tabel klaim
  beserta rujukan `file:baris`, dan status implementasi.
- [`docs/adr/0001-jalur-dekomposisi-pca.md`](docs/adr/0001-jalur-dekomposisi-pca.md):
  jalur dekomposisi (covariance_eigh vs full SVD).
- [`docs/adr/0002-batas-preprocessing.md`](docs/adr/0002-batas-preprocessing.md): batas
  tanggung jawab preprocessing (zestcore vs pemanggil).
- [`docs/proofs/pca_internals_check.py`](docs/proofs/pca_internals_check.py): skrip yang
  memeriksa setiap klaim di spesifikasi.

## Contoh pemakaian

`zestcore` menerima matriks yang sudah diproses seperti di Orange (Continuize, imputasi,
dan Normalize bila dipakai; lihat ADR 0002):

```python
import numpy as np
from zestcore.pca import pca

X = np.random.default_rng(0).normal(size=(200, 5))
res = pca(X)                      # k = min(n, p)
res.components                    # (k, p), satu baris per komponen
res.explained_variance_ratio      # (k,)
res.scores                        # (n, k)
```

Tanda tiap komponen dibuat deterministik: elemen dengan nilai mutlak terbesar selalu
positif, sama dengan konvensi `svd_flip` di scikit-learn.

## Menjalankan test

Semua angka dihasilkan dengan Python bawaan Orange supaya sama dengan GUI:

| | Versi |
|---|---|
| Orange | 3.40.0 |
| Python | 3.12.12 |
| NumPy | 2.1.3 |
| scikit-learn | 1.5.2 |

Dari root repo (Windows, PowerShell; sesuaikan `D:\Orange` dengan lokasi instalasi
Orange):

```powershell
D:\Orange\python.exe -m pip install -e . --no-deps
D:\Orange\python.exe -m pip install pytest
D:\Orange\python.exe -m pytest tests -q
```

Atau di lingkungan Python 3.12 biasa, dengan versi yang sama seperti di CI:

```bash
pip install numpy==2.1.3 scikit-learn==1.5.2 Orange3==3.40.0 PyQt5==5.15.11 PyQtWebEngine==5.15.7 pytest
pip install -e . --no-deps
QT_QPA_PLATFORM=offscreen pytest tests -q
```

### Data bank-full

Dataset [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
tidak disimpan di repo. Test yang memakainya membaca `bank-full.csv` dari path di env
`ZEST_BANK_FULL` dan otomatis di-skip bila file tidak ada. Di CI test ini selalu di-skip
(40 lulus, 7 skip); dengan datanya, semua 47 test lulus.

```powershell
$env:ZEST_BANK_FULL = "C:\path\ke\bank-full.csv"
```

Cara menjalankan ulang notebook ada di
[`evidence/pca/README.md`](evidence/pca/README.md#menjalankan-tes-dan-notebook).

## Lisensi

[GPL-3.0-or-later](LICENSE), sama dengan Orange.
