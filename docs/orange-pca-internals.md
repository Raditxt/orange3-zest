# Spesifikasi: cara Orange 3.40.0 menjalankan PCA

Lingkungan: `D:\Orange\python.exe` (Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2,
NumPy 2.1.3). Semua `file:baris` merujuk ke kode yang terpasang di
`D:\Orange\Lib\site-packages\`, bukan GitHub master. Skrip bukti:
`docs/proofs/pca_internals_check.py` (satu fungsi `check_klaimN()` per klaim).
Toleransi semua perbandingan: `np.allclose` default (rtol=1e-5, atol=1e-8) +
selisih maksimum dilaporkan.

## Status & langkah berikutnya

**Status (20 Nov 2026):** DRAF. Klaim 1-4 terbukti. Klaim 5-7 belum.
Task Notion: "Orange T1: baca kode sumber PCA Orange" (Lagi dikerjakan).

Langkah berikutnya, berurutan, satu langkah per giliran (tunggu Raditya):

1. **Klaim 5** (tanda `svd_flip(u_based_decision=False)`): baca `sklearn/utils/extmath.py`
   (`svd_flip`) dan pemanggilnya di `_pca.py`; buktikan di iris & heart_disease bahwa
   tiap baris `components_` punya elemen |terbesar| positif. Sekaligus cetak selisih
   eigenvalue berurutan komponen 1-18 heart_disease (cek tidak ada eigenvalue kembar
   di luar 7 komponen ~0), lalu tunjukkan `components_` 1-18 identik antara
   `covariance_eigh` dan `full`, sedangkan baris 19-25 (ruang nol) tidak.
2. **Klaim 6**: `PCA(n_components=100, random_state=0)` dan pemotongan
   `min(min(X.shape), n_components)` di `Orange/projection/pca.py:40-41`.
3. **Klaim 7**: heart_disease: jumlah NaN, jumlah variabel diskrit, kolom sebelum
   (13) / sesudah Continuize (25). Cocokkan jumlah variabel diskrit dengan 7 komponen
   bervarians ~0 di Klaim 4. Tunjukkan nilai indikator hasil imputasi (pecahan).
4. Lengkapi tabel + Implikasi di dokumen ini, ubah status jadi FINAL.
5. **ADR** via skill `domain-modeling`, bagian Keputusan DIKOSONGKAN (Raditya mengisi):
   - `docs/adr/0001-jalur-dekomposisi-pca.md`: "Jalur dekomposisi untuk inti PCA
     (covariance_eigh vs SVD)". Opsi, konsekuensi, data solver per dataset dari Klaim 4.
   - `docs/adr/0002-...md`: "Batas tanggung jawab preprocessing: di zestcore (NumPy)
     atau di pemanggil (Orange)". Opsi + konsekuensi.
6. Tambah satu baris di `CLAUDE.md`:
   "Spesifikasi PCA: docs/orange-pca-internals.md; keputusan desain: docs/adr/."
7. Setelah Raditya konfirmasi selesai: isi Catatan task Notion + kirim 📊 Laporan ke PM
   (aturan di `CLAUDE.md`).

Skill yang disarankan untuk sesi berikutnya: `andrej-karpathy-skills:karpathy-guidelines`
(langkah kecil, tiap klaim bisa dicek) dan `domain-modeling` (ADR). Jangan pakai skill
yang membuat issue/ticket atau agen latar belakang. Belum boleh menulis kode di `zestcore/`.

## Tabel klaim

| No | Klaim | Sumber (file:baris) | Bukti REPL | Hasil |
|----|-------|---------------------|------------|-------|
| 1 | `PCA.preprocessors = [Continuize(), SklImpute()]`; widget menambah `Normalize(center=False)` di akhir (default widget `normalize=True`) | `Orange/projection/base.py:268-269`; `Orange/projection/pca.py:27`; `Orange/widgets/unsupervised/owpca.py:50, 277-283` | `check_klaim1()`: `OWPCA._create_projector` dipanggil dengan objek tiruan | **Benar.** Default: `[Continuize, SklImpute]`; widget: `[Continuize, SklImpute, Normalize]`, `center=False`, `norm_type=NormalizeBySD` |
| 2 | Continuize default = Indicators, zero_based; tiap nilai diskrit jadi kolom 0/1, biner jadi 2 kolom | `Orange/preprocess/preprocess.py:47-48`; `Orange/preprocess/continuize.py:21-25, 45-60`; `Orange/preprocess/transformation.py:119-128, 152` | `check_klaim2()`: tabel mini 4×4 (umur, jk biner, warna 3 nilai, konstan 1 nilai, 1 NaN) | **Benar.** 4 kolom jadi 6: `umur, jk=L, jk=P, warna=merah, warna=hijau, warna=biru`. Tambahan: variabel diskrit dengan nilai kurang dari 2 dibuang; diskrit kosong menghasilkan NaN di semua kolom indikatornya |
| 3 | SD di Normalize ddof=0; kovarians PCA ddof=1 | `Orange/preprocess/normalize.py:25, 48, 53-58`; `Orange/statistics/util.py:352, 485` (`bn.nanvar`, ddof default 0); `sklearn/decomposition/_pca.py:591, 611-617` | `check_klaim3()` pada iris (150×4) | **Benar.** Normalize vs X/std(ddof=0): True, maks 1.78e-15; vs ddof=1: **False**, maks 0.0340. `explained_variance_` vs eig cov ddof=1: True, maks 6.01e-15; vs ddof=0: **False**, maks 0.0282 |
| 4 | `svd_solver='auto'` memilih `covariance_eigh` jika p ≤ 1000 dan n ≥ 10p | `sklearn/decomposition/_pca.py:531-543` (pemilihan), `611-643` (isi jalur; eigenvalue negatif dipotong ke 0 di `637`); `Orange/projection/pca.py:33` | `check_klaim4()`: projector widget (Normalize ON), baca `model.proj._fit_svd_solver` | **Benar.** iris 150×4 (10p=40): `covariance_eigh`. heart_disease 303×25 (10p=250): `covariance_eigh`. Bank marketing: belum dicek (data belum ada di repo) |
| 5 | Tanda: `svd_flip(u_based_decision=False)`, elemen |terbesar| tiap baris `components_` positif | belum | belum | **belum** |
| 6 | Widget `PCA(n_components=100, random_state=0)`, dipotong ke `min(X.shape)` | `owpca.py:27, 278`; `pca.py:40-41` (sudah dibaca, belum dibuktikan) | belum | **belum** |
| 7 | heart_disease punya NaN dan variabel diskrit | belum | belum | **belum** |

Catatan angka Klaim 4: `explained_variance_` heart_disease (jalur widget) punya **7 komponen
< 1e-10** (`3.9e-15, 1.4e-15, 8.0e-16, 5.7e-16, 0, 0, 0`; tiga terakhir hasil pemotongan ke 0).
Dugaan: satu arah bervarians nol per variabel diskrit (kolom indikator berjumlah 1);
dicocokkan di Klaim 7.

## Implikasi untuk zestcore

Urutan yang harus ditiru (jalur widget default):

1. **Continuize (Indicators, zero_based):** untuk tiap variabel diskrit dengan nilai ≥ 2,
   buat satu kolom 0/1 per nilai, **urut sesuai `var.values`**, termasuk variabel biner
   (2 kolom). Variabel diskrit dengan nilai < 2 dibuang. Kolom kontinu tidak diubah.
   Nilai diskrit kosong menghasilkan NaN di semua kolom indikator variabel itu (bukan 0).
   Kalau zestcore melakukan one-hot sendiri, urutan kolom dan aturan NaN ini wajib sama.
2. **SklImpute (mean):** NaN diganti mean kolom (`SimpleImputer(strategy='mean')`,
   `preprocess.py:150-169`). Kolom yang seluruhnya NaN dibuang. Dibuktikan dengan angka
   di Klaim 7.
3. **Normalize(center=False), hanya bila widget Normalize ON:** bagi tiap kolom dengan
   SD **ddof=0**, tanpa mengurangi mean. Kolom dengan SD = 0 atau NaN: SD diganti 1
   (kolom tidak dibagi).
4. **PCA:** pusatkan dengan mean kolom, kovarians **ddof=1**,
   `explained_variance_` = eigenvalue (menurun). Eigenvalue negatif dipotong ke 0
   (perilaku `covariance_eigh`). Jalur dekomposisi: menunggu ADR 0001.
5. **Test wajib membandingkan skor PC dan `explained_variance_`**, bukan hanya
   `components_`/rasio. Salah ddof di Normalize = skala seragam sqrt((n-1)/n), sehingga
   `components_` dan `explained_variance_ratio_` tetap sama (True), sedangkan
   `explained_variance_` bergeser tepat (n-1)/n = 0.99333 dan skor PC bergeser
   (maks 0.0110 di iris). Hanya test skor/varians yang menangkap kesalahan ini.
6. **Komponen bervarians ~0 tidak unik:** di heart_disease ada 7 eigenvalue ~0 (kembar).
   Vektor eigen di ruang itu bergantung pada LAPACK/urutan operasi, jadi `components_`
   baris 19-25 tidak bisa diharapkan identik. Test komponen dibatasi pada komponen
   dengan eigenvalue tidak kembar (bukti angka menyusul di Klaim 5).
7. Konvensi tanda (Klaim 5) dan jumlah komponen (Klaim 6): menyusul.
