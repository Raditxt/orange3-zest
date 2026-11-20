---
status: proposed
---

# Jalur dekomposisi inti PCA (covariance_eigh vs full SVD)

## Konteks

PCA Orange 3.40.0 memanggil sklearn `PCA(svd_solver='auto')`. Solver yang terpilih
bergantung pada bentuk X setelah preprocessing (`sklearn/decomposition/_pca.py:531-543`;
spesifikasi: `docs/orange-pca-internals.md`, Klaim 4 dan 6):

- p ≤ 1000 dan n ≥ 10p: `covariance_eigh` (eigen-dekomposisi matriks kovarians p × p).
  iris 150×4 dan heart_disease 303×25 masuk ke sini.
- Selain itu, bila min(n, p) ≤ 125: `full` (SVD dari X yang dipusatkan). Contoh:
  sintetis 1000×125.
- min(n, p) ≥ 126, n < 10p, max(n, p) > 500: `randomized`, hasilnya bergantung pada
  `random_state=0`. Contoh: sintetis 1000×126, 1000×200, 600×130.

Angka yang relevan:

- heart_disease, `covariance_eigh` vs `full`: komponen 1-18 `allclose` True
  (maks 2.98e-13), `explained_variance_` 1-18 juga True. Komponen 19-25 (ruang nol, 7
  eigenvalue ~0) False (maks 0.858), tapi itu tidak diuji (aturan test poin 2).
- `covariance_eigh` memotong eigenvalue negatif ke 0 (`_pca.py:637`); heart_disease
  punya 3 eigenvalue yang dipotong begitu.
- `random_state` 0 vs 12345 tidak mengubah hasil di `covariance_eigh` maupun `full`
  (`array_equal` True).
- Tanda keduanya ditentukan `svd_flip` berbasis baris yang sama (`_pca.py:647`).

## Opsi

1. **Hanya `covariance_eigh`.** Satu jalur, sama dengan pilihan `auto` untuk iris dan
   heart_disease. Kovarians mengkuadratkan bilangan kondisi, jadi eigenvalue kecil
   kurang akurat; untuk data n < 10p hasilnya dibandingkan dengan `full` milik Orange,
   dan kesamaannya baru terbukti di satu dataset (heart_disease).
2. **Hanya `full` SVD.** Satu jalur, lebih stabil secara numerik. Untuk iris dan
   heart_disease, Orange sendiri memakai `covariance_eigh`, jadi kesamaan bergantung pada
   selisih antarjalur (2.98e-13 di heart_disease, jauh di bawah toleransi).
3. **Meniru `auto` (dua jalur + aturan pemilihan).** Paling dekat dengan Orange untuk
   setiap bentuk data. Kode dan test dua kali lipat; aturan pemilihan sklearn harus
   ikut dijaga bila versi sklearn di Orange berubah.

Cakupan `randomized` (berlaku untuk opsi mana pun):

- **a. Batasi zestcore ke min(n, p) ≤ 125**, tolak atau beri peringatan di luar itu.
  Sederhana, cukup untuk iris dan heart_disease; bank marketing perlu dicek dulu.
- **b. Tiru `randomized` + `random_state=0`.** Harus mereproduksi urutan bilangan acak
  dan iterasi `randomized_svd` sklearn agar identik; jauh lebih rumit.

## Konsekuensi

- Opsi 1 atau 2: data yang di Orange jatuh ke jalur lain hanya *mendekati* identik,
  besar selisihnya harus diukur per dataset, bukan diasumsikan.
- Opsi 3: identik pada jalur yang sama, tapi zestcore ikut terikat pada aturan internal
  sklearn 1.5.2.
- Cakupan a: zestcore tidak bisa dipakai untuk data lebar (min(n, p) ≥ 126) dan n < 10p
  di Tahap 1.
- Komponen di ruang nol tidak identik di opsi mana pun; test membatasi diri pada
  komponen dengan eigenvalue tidak kembar.

## Keputusan

<!-- Diisi Raditya. -->
