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

Tahap 1 menulis ulang PCA, yang sebenarnya sudah ada di Orange. Tujuannya membuktikan
bahwa algoritma bisa ditulis ulang di luar Orange dan hasilnya dapat dibuktikan sama,
sebelum masuk ke model yang belum ada di Orange. Hasilnya dibungkus menjadi widget
**Zest PCA**, yang juga menjadi template untuk widget model berikutnya.

## Memasang add-on

Add-on belum ada di PyPI. Pasang dari folder repo ke **Python bawaan Orange** (sesuaikan
`D:\Orange` dengan lokasi instalasi Orange):

```powershell
D:\Orange\python.exe -m pip install -e .
```

Tutup lalu buka lagi Orange. Di panel widget muncul kategori **Zest** berisi **Zest PCA**
(screenshot: [`evidence/pca/zest_category.png`](evidence/pca/zest_category.png)).

## Widget Zest PCA

Membungkus `zestcore.pca` (NumPy murni) dengan input dan output yang sama dengan widget
PCA Orange:

- Input `Data`.
- Output `Transformed Data`: skor PC1..PCk, dengan kelas dan meta asli ikut dibawa. Tiap PC
  membawa rasio variansnya. Domain hasil bisa dipakai Apply Domain untuk data baru.
- Output `Components`: satu baris per komponen, satu kolom per fitur, ditambah meta
  `components` dan `variance`.
- Setting: Components (1-100, default 2) dan Normalize variables (default ON).
  Preprocessing (imputasi, Continuize, Normalize) dikerjakan preprocessor Orange, sama
  dengan widget PCA ([ADR 0002](docs/adr/0002-batas-preprocessing.md),
  [ADR 0003](docs/adr/0003-output-widget-dan-transform.md)).

**Klaim:** hasil Zest PCA identik dengan PCA Orange 3.40.0 **dalam toleransi spec**
(`np.allclose`, rtol=1e-5, atol=1e-8; aturan komponen seri dan ruang nol di
[spesifikasi](docs/orange-pca-internals.md#aturan-test)) **untuk X yang diproses
preprocessor Orange**. Klaim ini diuji di `tests/test_owzestpca.py` (widget vs widget)
dan dibuktikan di GUI pada kasus heart_disease. Selisih maks skor 8.88e-16, komponen
25 × 25 dan skor 303 × 25, lihat
[`evidence/pca/`](evidence/pca/README.md#bukti-gui-t2a-zest-pca-vs-pca-orange-heart_disease-27-nov-2026).

**Batas bentuk data ([ADR 0001](docs/adr/0001-jalur-dekomposisi-pca.md)):** zestcore hanya
meniru jalur `covariance_eigh`, yang dipakai Orange bila p ≤ 1000 dan n ≥ 10p (n baris,
p kolom setelah preprocessing). Di luar batas itu Orange memakai SVD `full` atau
`randomized`, sehingga hasil Zest PCA hanya mendekati dan tidak dijamin identik. Widget
menampilkan warning kuning dengan pesan itu, tetapi output tetap dikirim.

**Keterbatasan yang diketahui:**

- Belum ada scree plot (grafik explained variance). Direncanakan sebagai task lanjutan.
- Belum ada spin "explained variance", output `Data` dan `PCA` (projector), maupun tombol
  auto-apply seperti di widget PCA Orange.
- Fit yang gagal (misalnya kurang dari 2 baris, atau tidak ada atribut setelah
  preprocessing) tampil sebagai error merah dengan pesan mentah dari NumPy/scikit-learn.
- Fit dijalankan sinkron (tanpa thread), cukup cepat untuk data seukuran bank-full
  (45211 × 51).

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
| `orangecontrib/zest/` | Lapis Orange yang hanya membungkus `zestcore`: adaptor `pca.py` (projector Orange) dan `widgets/` (Zest PCA). |
| `tests/` | pytest. Orange dan scikit-learn dipakai di sini sebagai pembanding. |
| `notebooks/` | Notebook bukti: `01_pca_gui_evidence` (GUI vs skrip), `02_pca_perbandingan` (zestcore, scikit-learn, GUI vs Orange). |
| `evidence/pca/` | Output widget PCA dan Zest PCA yang direkam dari GUI Orange (`.pkl`, `.npy`, CSV, `.ows`, screenshot). Lihat [README-nya](evidence/pca/README.md). |
| `docs/` | Spesifikasi PCA Orange dan keputusan desain (ADR). |

### Dokumentasi

- [`docs/orange-pca-internals.md`](docs/orange-pca-internals.md): spesifikasi cara Orange
  3.40.0 menjalankan PCA, langkah yang ditiru, aturan test, batas cakupan, tabel klaim
  beserta rujukan `file:baris`, dan status implementasi.
- [`docs/adr/0001-jalur-dekomposisi-pca.md`](docs/adr/0001-jalur-dekomposisi-pca.md):
  jalur dekomposisi (covariance_eigh vs full SVD).
- [`docs/adr/0002-batas-preprocessing.md`](docs/adr/0002-batas-preprocessing.md): batas
  tanggung jawab preprocessing (zestcore vs pemanggil).
- [`docs/adr/0003-output-widget-dan-transform.md`](docs/adr/0003-output-widget-dan-transform.md):
  output widget dan transform data baru (adaptor projector Orange).
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
(62 lulus, 9 skip); dengan datanya, semua 71 test lulus.

Lint (aturan di `pyproject.toml`, `[tool.ruff.lint]`):

```powershell
ruff check .
```

```powershell
$env:ZEST_BANK_FULL = "C:\path\ke\bank-full.csv"
```

Cara menjalankan ulang notebook ada di
[`evidence/pca/README.md`](evidence/pca/README.md#menjalankan-tes-dan-notebook).

## Lisensi

[GPL-3.0-or-later](LICENSE), sama dengan Orange.
