# Spesifikasi: cara Orange 3.40.0 menjalankan PCA

**Status: FINAL (20 Nov 2026).** Keputusan desain yang bergantung pada spesifikasi ini:
`docs/adr/0001-jalur-dekomposisi-pca.md`, `docs/adr/0002-batas-preprocessing.md`.

Lingkungan: `D:\Orange\python.exe` (Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2,
NumPy 2.1.3). Semua `file:baris` merujuk ke kode yang terpasang di
`D:\Orange\Lib\site-packages\`, bukan GitHub master. Skrip bukti:
`docs/proofs/pca_internals_check.py` (satu fungsi `check_klaimN()` per klaim).
Toleransi semua perbandingan: `np.allclose` default (rtol=1e-5, atol=1e-8) +
selisih maksimum dilaporkan.

## Langkah yang ditiru zestcore (jalur widget default)

Klaim Tahap 1: PCA NumPy murni di zestcore identik dengan PCA Orange untuk X yang sudah
diproses preprocessor Orange. Langkah 1-3 dikerjakan **pemanggil** (lapis widget dan
test) dengan preprocessor Orange sendiri; zestcore hanya menerima X numerik siap pakai
dan mengerjakan langkah 4-7 (ADR 0002). Langkah 1-3 dicatat di sini supaya pemanggil
dan test tahu apa yang terjadi sebelum X sampai ke zestcore.

1. **Continuize** (Klaim 2). Variabel diskrit dengan nilai deklarasi ≥ 2 jadi satu
   kolom 0/1 per nilai, urut sesuai `var.values`; biner juga 2 kolom. Nilai deklarasi
   < 2: variabel dibuang (`continuize.py:21`). Nilai kosong: NaN di semua kolom
   indikator variabel itu. Kolom kontinu tidak diubah.
2. **Imputasi mean** (Klaim 2, 7). NaN diganti mean kolom; kolom yang seluruhnya NaN
   dibuang (`preprocess.py:150-173`). Indikator yang diimputasi jadi pecahan
   (heart_disease, `thal`: `[0.5515, 0.3887, 0.0598]`), jumlahnya tetap 1.
3. **Normalize, hanya bila opsi Normalize ON** (default widget ON; Klaim 1, 3). Bagi
   tiap kolom dengan SD **ddof=0**, tanpa mengurangi mean. SD = 0 atau NaN diganti 1
   (`normalize.py:49-54`).
4. **Jumlah komponen + penjaga bentuk** (Klaim 4, 6; ADR 0001). Widget: k = min(n, p,
   100), n dan p dihitung *setelah* langkah 1-3. Pilihan jumlah komponen di widget hanya
   mengiris output. zestcore memotong ke min(n, p); angka 100 dikirim pemanggil
   (`n_components=100`), tidak ditetapkan di zestcore. Bila X tidak memenuhi p ≤ 1000 dan n ≥ 10p, zestcore tetap
   menghitung tetapi mengeluarkan `warnings.warn`: Orange memakai jalur lain (`full`
   atau `randomized`) untuk bentuk ini, jadi hasilnya hanya mendekati dan tidak dijamin
   identik.
5. **Fit** (Klaim 3, 4, 6). Kovarians **ddof=1**, dihitung persis dalam bentuk dan
   urutan perkalian sklearn: C = X.T @ X, C −= (n·mean)·meanᵀ, C /= n−1
   (`_pca.py:611-617`). Setara matematis dengan Xc.T @ Xc/(n−1) tapi hasil
   floating-point berbeda; urutan (n·a)·b juga diikuti karena n·(a·b) mengubah digit
   terakhir C. Dengan bentuk ini C sama persis dengan milik Orange;
   eigen-dekomposisi kovarians (hanya jalur `covariance_eigh`, ADR 0001); urutkan
   eigenvalue menurun; eigenvalue negatif dipotong ke 0. `explained_variance_` = eigenvalue. `explained_variance_ratio_` =
   eigenvalue / jumlah *semua* min(n, p) eigenvalue sebelum diiris ke k
   (`_pca.py:652-653`); bila k < min(n, p), jumlah rasio < 1.
6. **Tanda** (Klaim 5). Untuk tiap baris `components_`, cari elemen |terbesar|
   (indeks pertama bila seri) dan kalikan baris dengan tandanya (`svd_flip`
   berbasis baris, `extmath.py:848-905`).
7. **Skor**: X @ `components_`.T − mean @ `components_`.T, bentuk sklearn `transform`
   (`decomposition/_base.py:149, 156`, whiten=False), dipanggil Orange lewat
   `Orange/projection/base.py:115-116`. Tidak
   punya klaim tersendiri; dibuktikan lewat test skor.

## Aturan test

1. **Bandingkan skor PC dan `explained_variance_`**, bukan hanya `components_`/rasio.
   Salah ddof di Normalize menghasilkan skala seragam sqrt((n-1)/n): `components_` dan
   rasio tetap sama, sedangkan `explained_variance_` bergeser (n-1)/n = 0.99333 dan skor
   bergeser maks 0.0110 (iris, Klaim 3).
2. **Komponen di ruang nol tidak dibandingkan.** Eigenvalue ~0 kembar, jadi vektornya
   tidak unik: heart_disease komponen 1-18 `covariance_eigh` vs `full` True
   (maks 2.98e-13), 19-25 False (maks 0.858). Jumlah arah varians nol
   (eigenvalue < 1e-10) diprediksi di sisi test dari data sebelum fit (butuh metadata
   variabel Orange, ADR 0002):

   jumlah nol = Σ (nilai deklarasi − nilai yang muncul + 1) untuk tiap variabel
   diskrit yang tidak dibuang, + jumlah kolom kontinu konstan.

   Bila semua nilai deklarasi muncul, suku diskrit = 1 per variabel. Variabel yang
   dibuang (nilai deklarasi < 2, atau seluruhnya NaN) menyumbang 0. Rumus ini batas
   bawah: arah nol tambahan muncul bila n − 1 < p atau ada kolom yang kolinear
   (termasuk dua kolom indikator dari variabel berbeda yang selalu muncul bersama).
   Terbukti di heart_disease (7 = 7) dan data sintetis (6 = 6; rumus tanpa suku
   "nilai yang tidak muncul" memprediksi 4). Bank marketing (UCI): `bank-full` 9 = 9;
   `bank-additional-full` 11 vs prediksi 10, tambahannya dari `housing=unknown` dan
   `loan=unknown` yang muncul di 990 baris yang sama persis.

   Skor ruang nol tetap dibandingkan, dengan rtol=0 dan atol=1e-8 (nilainya noise
   ~1e-15, jadi galat relatif tidak bermakna). Aturan yang sama dipakai untuk rasio
   varians ruang nol (~1e-16) dan `explained_variance_` ruang nol (~1e-15, sebagian
   dipotong ke 0).
3. **Tanda komponen seri** (disetujui Raditya, 20 Nov 2026). Baris k acuan (Orange)
   disebut seri bila |terbesar| − |kedua terbesar| < 1e-10.
   - Tidak seri: `components_[k]` dan kolom skor ke-k dibandingkan langsung.
   - Seri: s = sign(a_k · b_k) dari komponen (a = acuan, b = zestcore), lalu bandingkan
     `a_k` dengan `s·b_k` **dan** skor acuan ke-k dengan `s·` skor zestcore ke-k
     (bukan diuji terpisah dengan "or").
   - Toleransi tetap `np.allclose` default.
   - Baris seri juga muncul di ruang nol heart_disease (PC19, 22, 24, 25); tidak
     relevan karena komponen ruang nol tidak dibandingkan (poin 2).

## Batas cakupan yang diketahui

- **Komponen seri dari variabel biner.** Continuize mengubah variabel biner jadi `x`
  dan `1-x`; setelah dinormalisasi loading keduanya sama besar beda tanda, sehingga
  `argmax` diputuskan galat ~1e-16. heart_disease: PC2 (`gender`) dan PC3
  (`fasting blood sugar > 120`); iris: 0 dari 4; bank-full: PC2, 4, 6, 7 (di luar
  ruang nol, terukur 23 Nov 2026). Acak urutan baris 20×: tanda PC2
  terbalik 10×, PC3 2×, di Orange sendiri.
- **Jalur `randomized`.** Bila min(n, p) ≤ 125, `auto` hanya memilih `covariance_eigh`
  atau `full`, dan `random_state` tidak berpengaruh. Bila min(n, p) ≥ 126, k = 100 <
  0.8·min(n, p) dan `auto` memilih `randomized` bila n < 10p dan max(n, p) > 500
  (`_pca.py:539`; sintetis 1000×126, 1000×200, 600×130). `randomized` tidak ditiru;
  di luar syarat `covariance_eigh` selisihnya diukur, bukan diasumsikan (ADR 0001).
- **Bank marketing** (UCI, dicek 20 Nov 2026, data tidak disimpan di repo; target `y`
  tidak ikut, "unknown" adalah kategori, bukan NaN): `bank-full` 45211×16 → 45211×51,
  `bank-additional-full` 41188×20 → 41188×63 setelah Continuize. Keduanya n ≥ 10p,
  `auto` memilih `covariance_eigh`, jadi masuk cakupan ADR 0001.

## Tabel klaim

| No | Klaim | Sumber (file:baris) | Bukti REPL | Hasil |
|----|-------|---------------------|------------|-------|
| 1 | `PCA.preprocessors = [Continuize(), SklImpute()]`; widget menambah `Normalize(center=False)` di akhir (default widget `normalize=True`) | `Orange/projection/base.py:268-269`; `Orange/projection/pca.py:27`; `Orange/widgets/unsupervised/owpca.py:50, 277-283` | `check_klaim1()`: `OWPCA._create_projector` dipanggil dengan objek tiruan | **Benar.** Default: `[Continuize, SklImpute]`; widget: `[Continuize, SklImpute, Normalize]`, `center=False`, `norm_type=NormalizeBySD` |
| 2 | Continuize default = Indicators, zero_based; tiap nilai diskrit jadi kolom 0/1, biner jadi 2 kolom | `Orange/preprocess/preprocess.py:47-48`; `Orange/preprocess/continuize.py:21-25, 45-60`; `Orange/preprocess/transformation.py:119-128, 152` | `check_klaim2()`: tabel mini 4×4 (umur, jk biner, warna 3 nilai, konstan 1 nilai, 1 NaN) | **Benar.** 4 kolom jadi 6: `umur, jk=L, jk=P, warna=merah, warna=hijau, warna=biru`. Tambahan: variabel diskrit dengan nilai kurang dari 2 dibuang; diskrit kosong menghasilkan NaN di semua kolom indikatornya |
| 3 | SD di Normalize ddof=0; kovarians PCA ddof=1 | `Orange/preprocess/normalize.py:25, 48, 53-58`; `Orange/statistics/util.py:352, 485` (`bn.nanvar`, ddof default 0); `sklearn/decomposition/_pca.py:591, 611-617` | `check_klaim3()` pada iris (150×4) | **Benar.** Normalize vs X/std(ddof=0): True, maks 1.78e-15; vs ddof=1: **False**, maks 0.0340. `explained_variance_` vs eig cov ddof=1: True, maks 6.01e-15; vs ddof=0: **False**, maks 0.0282 |
| 4 | `svd_solver='auto'` memilih `covariance_eigh` jika p ≤ 1000 dan n ≥ 10p | `sklearn/decomposition/_pca.py:531-543` (pemilihan), `611-643` (isi jalur; eigenvalue negatif dipotong ke 0 di `637`); `Orange/projection/pca.py:33` | `check_klaim4()`: projector widget (Normalize ON), baca `model.proj._fit_svd_solver` | **Benar.** iris 150×4 (10p=40): `covariance_eigh`. heart_disease 303×25 (10p=250): `covariance_eigh`, 7 komponen < 1e-10 (`3.9e-15, 1.4e-15, 8.0e-16, 5.7e-16, 0, 0, 0`; tiga terakhir hasil pemotongan ke 0, dijelaskan Klaim 7). bank-full 45211×51 (10p=510): `covariance_eigh` (`notebooks/02_pca_perbandingan.ipynb`, tabel ringkasan) |
| 5 | Tanda: `svd_flip(u_based_decision=False)`, elemen \|terbesar\| tiap baris `components_` positif | `sklearn/utils/extmath.py:848-905` (`svd_flip`; cabang `else` = berbasis baris `v`, `argmax` ambil indeks pertama bila seri); `sklearn/decomposition/_pca.py:647` (dipanggil untuk jalur `full` dan `covariance_eigh`), `760, 773` (jalur arpack/randomized). Orange tidak membalik tanda sendiri (grep `sign\|flip` di `pca.py`, `base.py`, `owpca.py`: nihil) | `check_klaim5()`: jalur widget di iris & heart_disease; `covariance_eigh` vs `full`; deteksi baris seri; acak urutan baris 20× | **Benar, dengan catatan.** Semua baris positif (iris 4/4, heart_disease 25/25). Komponen 1-18 heart_disease: selisih eigenvalue terkecil 0.0127 (tidak kembar); `eigh` vs `full`: True, maks 2.98e-13. Baris 19-25: False, maks 0.858. **Tapi** PC2 dan PC3 heart_disease *seri* (jarak 6.7e-16, 8.9e-16): dua kolom indikator variabel biner (`gender`, `fasting blood sugar > 120`) punya loading sama besar beda tanda. Acak urutan baris 20×: tanda PC2 terbalik 10×, PC3 2× |
| 6 | Widget `PCA(n_components=100, random_state=0)`, dipotong ke `min(X.shape)` | `Orange/widgets/unsupervised/owpca.py:27` (`MAX_COMPONENTS = 100`), `278-279` (projector; `ncomponents` hanya disimpan di `projector.component`), `176-186` (fit sekali, ratio dari model), `326-354` (output hanya diiris `[:self.ncomponents]`); `Orange/projection/pca.py:38-41` (pemotongan, X setelah preprocessing); `sklearn/decomposition/_pca.py:738, 754, 771` (`random_state` hanya dipakai arpack/randomized), `698-705` (pemotongan sklearn) | `check_klaim6()`: parameter projector; iris, heart_disease, iris 3 baris; sklearn tanpa pemotongan; `ncomponents` 2 vs 5; `random_state` 0 vs 12345 | **Benar.** iris 150×4 → 4 komponen; heart_disease 303×25 → 25 (dari 25 kolom sesudah Continuize, bukan 13); iris 3 baris 3×4 → 3 (dipotong oleh n). Tanpa pemotongan sklearn `ValueError`. `sum(ratio)` = 1.000000000000000 di ketiganya. `ncomponents` 2 vs 5: `components_` sama persis (`array_equal`). `random_state` 0 vs 12345: sama persis di `covariance_eigh` dan `full`. **Batas:** pemotongan sebenarnya `min(n, p, 100)`; data sintetis 1000×125 → `full`, 100 komponen; 1000×126, 1000×200, 600×130 → **`randomized`**, 100 komponen (`_pca.py:539`) |
| 7 | heart_disease: 6 NaN, 7 variabel diskrit; 13 → 25 kolom; 7 arah varians ~0 = 1 per variabel diskrit | `Orange/preprocess/preprocess.py:150-173` (SklImpute mean, kolom seluruhnya NaN dibuang); `Orange/preprocess/continuize.py:21, 45-60`; `Orange/preprocess/transformation.py:131-152`; `Orange/preprocess/normalize.py:49-54` (SD 0/NaN → 1) | `check_klaim7()`: tipe & NaN per atribut; kolom per tahap; indikator hasil imputasi; arah w per grup indikator; data sintetis 50 baris | **Benar.** NaN: `major vessels colored` 4, `thal` 2 (6 baris). 7 diskrit + 6 kontinu; 13 → 25 kolom = 6 + (2+4+2+3+2+3+3). Indikator `thal` baris 88, 267 = `[0.5515, 0.3887, 0.0598]` (mean kolom), jumlah per grup = 1 (maks selisih 0). Arah w (w_j = sd_j di kolom grup): ke komponen 1-18 maks 4.4e-15, sisa di 19-25 maks 4.3e-15, rank 7, varians 0. Sintetis (kontinu konstan, nilai tidak muncul, diskrit seluruhnya NaN, diskrit 1 nilai): 6 arah nol, rumus koreksi 6, rumus awal 4 |

## Status implementasi (handoff 23 Nov 2026, sore)

Bagian ini untuk sesi berikutnya setelah `/clear`: cukup baca ini + CLAUDE.md.

### Selesai

- **Inti PCA NumPy murni** (`1503b04`). `zestcore/pca.py`: `pca(X, n_components=None)` →
  `PCAResult(mean, components, explained_variance, explained_variance_ratio, scores)`.
  Langkah 4-7, jalur `covariance_eigh` saja (ADR 0001); X harus sudah diproses
  preprocessor Orange (ADR 0002). Widget memakai `n_components=100`, dikirim pemanggil.
  Belum ada `transform` untuk data baru (ditambah saat widget dibuat). Warning di luar
  p ≤ 1000 dan n ≥ 10p; `ValueError` bila n < 2.
- **Kovarians urutan persis sklearn** (`d8f4fe9`). `n * np.outer(mean, mean)` diganti
  `n * mean[:, None] * mean[None, :]` = (n·a)·b. Hasilnya komponen, `explained_variance_`,
  dan rasio zestcore sama persis dengan Orange (0.0) di ketiga dataset.
- **Bukti GUI bank-full** (`b27afbd`). `evidence/pca/bank_pca_norm-on_51pc_*` (`.npy`,
  `.pkl`, CSV Components; tanpa CSV skor) + `.ows` + screenshot. `.ows` sudah disimpan
  ulang: kedua Save Data anotasi OFF, `stored_name` sesuai file di repo.
- **Notebook perbandingan** (`4be7154`). `notebooks/02_pca_perbandingan.ipynb`: zestcore,
  scikit-learn, dan GUI vs PCA Orange (projector widget, Normalize ON) untuk iris,
  heart_disease, bank-full; tabel selisih maks, komponen seri, arah nol, sel bukti
  layout memori, kesimpulan bank-full, narasi BAB IV. Notebook ini dibangun dengan
  helper `tests/` (`compare`, `compare_variance`, `compare_ratio`, `preprocessed_X`,
  `load_table`), tanpa logika uji baru.
- Test: `D:\Orange\python.exe -m pytest tests -q -p no:warnings` → 47 lulus, 0 skip.
  bank-full dibaca dari `../magang-datamining/datasets/bank-full.csv` atau env
  `ZEST_BANK_FULL`; bila tidak ada, test bank di-skip.
- **GitHub Actions untuk test PCA** (`3ae50a7`). `.github/workflows/tests.yml`:
  ubuntu-latest, Python 3.12, versi paket sama dengan Python bawaan Orange, Qt
  `offscreen`. Run pertama di main hijau: 40 lulus, 7 skip (semua test bank-full,
  datanya tidak ada di repo).

Selisih maksimum vs PCA Orange (projector widget, Normalize ON, `n_components=100`).
Semua lolos aturan uji (non-nol rtol=1e-5 atol=1e-8; ruang nol rtol=0 atol=1e-8):

| Dataset | Pembanding | komponen | skor | skor nol | expl. var | rasio |
|---|---|---|---|---|---|---|
| iris (150×4, nol 0) | zestcore | 0.0 | 5.55e-17 | – | 0.0 | 0.0 |
| | scikit-learn | 0.0 | 0.0 | – | 0.0 | 0.0 |
| | GUI | 0.0 | 0.0 | – | – | 4.79e-7 (CSV %g) |
| heart_disease (303×25, nol 7) | zestcore | 0.0 | 8.88e-16 | 4.44e-16 | 0.0 | 0.0 |
| | scikit-learn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | GUI | 0.0 | 0.0 | 0.0 | – | 3.58e-7 (CSV %g) |
| bank-full (45211×51, nol 9) | zestcore | 0.0 | 3.11e-15 | 1.33e-15 | 0.0 | 0.0 |
| | scikit-learn | 0.0 | 1.78e-15 | 8.88e-16 | 0.0 | 0.0 |
| | GUI | 0.0 | 0.0 | 0.0 | – | 4.85e-8 (CSV %g) |

Komponen seri: heart_disease PC2, 3; bank-full PC2, 4, 6, 7 (s = +1 di semua
pembanding). Jumlah arah nol rumus = terukur untuk ketiganya.

Keputusan yang disetujui Raditya (jangan diubah tanpa diskusi):

- Kovarians bentuk dan urutan perkalian persis sklearn (lihat langkah 5).
- Skor bentuk sklearn X@C.T − mean@C.T (`_base.py:149, 156`). Komponen zestcore tetap
  C-order. Selisih skor zestcore vs sklearn (≤1.8e-15) berasal dari layout memori
  komponen (sklearn F-order) yang membuat BLAS mengambil jalur berbeda; dengan salinan
  F-order selisihnya 0.0 (sel bukti di notebook 02). Ini detail BLAS, bukan metode,
  jadi tidak ditiru.
- Test tetap memakai toleransi spec, tidak meng-assert bit-identik (bergantung BLAS).

### Pertanyaan terbuka

- bank-full: skor model Orange (`model(table)`, jalur `Table.transform`) berbeda 1.78e-15
  dari `proj.transform(X)` sklearn pada X yang sama (iris, heart_disease: 0.0). Ini
  sumber selisih skor scikit-learn vs Orange di tabel. Belum diselidiki; jauh di bawah
  toleransi.

### Task berikutnya

1. **Laporan Tahap 1 (BAB I–V)**, ditulis Raditya, dikerjakan di luar repo. Bahan BAB IV:
   tabel di atas dan narasi di akhir notebook 02. Engineer hanya membantu kerangka dan
   cek akurasi teknis.
2. **Tahap 2: kajian model kandidat** (task Notion "Orange T2: kajian model kandidat +
   pilih model pertama yang masuk add-on"). Memilih model pertama dari daftar rujukan
   Prof. Irfan; brief lengkap ada di Catatan task tersebut.

### Langkah lanjut (di luar Tahap 1)

- **bank-additional-full** (41188×20 → 41188×63 setelah Continuize). Baru dicek bentuk
  dan arah nolnya (11 vs prediksi 10, lihat Aturan test poin 2); belum dibandingkan dengan
  zestcore, scikit-learn, maupun GUI. Tahap 1 cukup dengan iris, heart_disease, dan
  bank-full.
