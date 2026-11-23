# Bukti GUI: PCA Orange 3.40.0

Output widget PCA yang direkam dari GUI Orange, dipakai untuk membuktikan bahwa PCA
Orange lewat skrip (projector widget) sama dengan yang dihasilkan GUI. Perbandingannya
ada di `notebooks/01_pca_gui_evidence.ipynb` (iris, heart_disease),
`notebooks/02_pca_perbandingan.ipynb` (ketiga dataset) dan `tests/` (aturan uji:
`docs/orange-pca-internals.md`).

## Lingkungan perekaman

| | Versi |
|---|---|
| Orange | 3.40.0 (GUI dan Python bawaan Orange) |
| Python | 3.12.12 |
| NumPy | 2.1.3 |
| scikit-learn | 1.5.2 |
| OS | Windows 11 |

Paket tambahan di Python Orange, hanya untuk tes dan notebook (tidak dipakai GUI):

| Paket | Versi |
|---|---|
| pytest | 9.2.0 |
| nbclient | 0.11.0 |
| nbformat | 5.11.1 |

## Workflow dan setting widget

Ketiga workflow sama bentuknya (`*_workflow.png`): File → PCA → Save Data (Transformed
Data), Save Data (1) (Components), Python Script (Components). Setting dibaca dari
`.ows`:

| Widget | Setting | iris | heart_disease | bank-full |
|---|---|---|---|---|
| File | dataset | `Orange/datasets/iris.tab` (bawaan) | `Orange/datasets/heart_disease.tab` (bawaan) | `bank-full.csv` (UCI, di luar repo), `y` = target |
| PCA | Normalize variables | ON | ON | ON |
| PCA | Components | 4 | 25 | 51 |
| PCA | Explained variance | 100% | 100% | 100% |
| PCA | Apply Automatically | ON | ON | ON |
| Save Data, Save Data (1) | Add type annotations | OFF | OFF | OFF |
| Save Data, Save Data (1) | Autosave | OFF | OFF | OFF |

Components = jumlah komponen maksimum untuk data ini (min(n, p)), jadi tidak ada
komponen yang terpotong. Setting PCA juga terlihat di `*_settings.png`.

## Daftar file

Nama file: `<data>_pca_norm-on_<k>pc_<output>.<ext>`.

| File | Output PCA | Ditulis oleh | Isi |
|---|---|---|---|
| `*_scores.pkl` | Transformed Data | Save Data | `Table` Orange: skor PC1..PCk + kelas (presisi penuh) |
| `*_scores.csv` | Transformed Data | Save Data | kelas, PC1..PCk; tanpa baris anotasi tipe; angka `%g` (6 digit signifikan) |
| `*_scores.csv.metadata` | Transformed Data | Save Data | atribut tabel (nama, deskripsi, referensi dataset), ditulis otomatis di samping CSV |
| `*_components.npy` | Components | Python Script | `components_` k × p, float64, presisi penuh |
| `*_components.csv` | Components | Save Data (1) | `components`, `variance` (rasio varians), lalu loading per fitur; `%g` |
| `*_pca.ows` | – | Orange | workflow lengkap dengan setting widget |
| `*_settings.png` | – | screenshot | jendela widget PCA |
| `*_workflow.png` | – | screenshot | kanvas workflow |

`<data>` = `iris` (150 × 4, k = 4), `heart` (heart_disease 303 × 25 setelah
Continuize, k = 25) dan `bank` (bank-full 45211 × 51 setelah Continuize, k = 51).
Untuk bank hanya ada `*_scores.pkl`; CSV skor (45211 baris) tidak dibuat.

## Catatan

- **bank-full tidak disimpan di repo** (lihat `docs/orange-pca-internals.md`, batas
  cakupan). File widget membaca `bank-full.csv` dengan `y` sebagai target, sehingga 16
  atribut menjadi 51 kolom. Tes dan notebook membacanya lewat `tests/datasets.py`
  (`../magang-datamining/datasets/bank-full.csv` atau env `ZEST_BANK_FULL`).
- **Save Data gagal menulis `.pkl` untuk output Components.** File yang dihasilkan
  0 byte. Karena itu komponen presisi penuh diambil lewat widget Python Script:

  ```python
  import numpy as np
  np.save(r"<repo>\evidence\pca\<data>_pca_norm-on_<k>pc_components.npy", in_data.X)
  print(in_data.X.shape, in_data.X.dtype)
  out_data = in_data
  ```

  File 0 byte itu tidak disimpan di repo. Skrip ini tersimpan di `.ows` sebagai
  `scriptText` node Python Script, bukan di library skrip widget (library hanya berisi
  template "Table from numpy"). Buka `.ows` di Orange untuk melihatnya.
- **`.ows` hanya mencatat satu nama file per Save Data.** Node Save Data (skor)
  menyimpan `stored_name` terakhir saja (iris: `.pkl`, heart: `.csv`), padahal kedua
  format skor ada untuk kedua dataset.
- **GUI tidak mengeluarkan `explained_variance_`**, hanya rasionya
  (`explained_variance_ratio_`, kolom `variance` di `*_components.csv`). Varians dari
  GUI hanya bisa diuji lewat rasio itu; `explained_variance_` sendiri diuji di skrip.
- **Acuan skrip memakai API privat Orange.** `OWPCA._create_projector` (Orange 3.40.0)
  dipakai untuk membuat projector yang sama persis dengan widget; bisa berubah di
  versi Orange lain.
- **CSV = pembulatan `%g`.** Dibanding `.pkl`/`.npy`, selisih relatif maks ≤ 5e-6
  (rtol=1e-5, atol=0). Untuk perbandingan presisi penuh pakai `.pkl`/`.npy`.
- **GUI vs skrip.** Di mesin perekaman, `.pkl`/`.npy` sama persis dengan PCA Orange
  lewat skrip (selisih maks 0.0). Itu observasi, bukan syarat; syarat tesnya
  `np.allclose` (rtol=1e-5, atol=1e-8), karena di mesin lain (BLAS berbeda) bisa
  muncul selisih ~1e-16.

## Menjalankan tes dan notebook

Dari root repo, di PowerShell, dengan Python bawaan Orange:

```powershell
# sekali saja: paket tes dan notebook
D:\Orange\python.exe -m pip install pytest nbclient nbformat

# tes
D:\Orange\python.exe -m pytest tests -q

# eksekusi ulang notebook (output disimpan di file yang sama)
$env:PYTHONUTF8 = "1"
D:\Orange\Scripts\jupyter-execute.exe --inplace notebooks\01_pca_gui_evidence.ipynb
```

CI (`.github/workflows/tests.yml`, GitHub Actions, ubuntu-latest) menjalankan tes yang
sama dengan versi paket di atas dan Qt `offscreen`. bank-full tidak ada di runner, jadi
tes bank di-skip di CI; tes bank hanya jalan di mesin lokal yang punya datanya.

Sesuaikan `D:\Orange` dengan lokasi instalasi Orange. `PYTHONUTF8=1` wajib di
Windows: tanpa itu `jupyter-execute` membaca notebook sebagai cp1252 dan merusak
karakter non-ASCII (`≤` jadi `â‰¤`). `python -m jupyter execute` tidak berjalan di
instalasi ini (keluar dengan kode 1 tanpa pesan); pakai `jupyter-execute.exe`.
