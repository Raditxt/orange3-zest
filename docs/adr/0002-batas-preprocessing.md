---
status: accepted
---

# Batas tanggung jawab preprocessing (zestcore vs pemanggil)

## Konteks

Sebelum PCA, widget Orange menjalankan Continuize → SklImpute → Normalize
(`docs/orange-pca-internals.md`, langkah 1-3, Klaim 1-3 dan 7). Langkah-langkah ini
mengubah data secara besar: heart_disease 13 kolom jadi 25, 6 NaN diimputasi, dan
indikator hasil imputasi jadi pecahan (`thal`: `[0.5515, 0.3887, 0.0598]`). Aturan
`CLAUDE.md`: `zestcore/` NumPy murni, dilarang `import Orange`/`sklearn`;
`orangecontrib/zest/` hanya membungkus `zestcore`.

Detail yang harus sama persis bila preprocessing ditiru:

- Continuize butuh metadata variabel: tipe kolom dan daftar nilai *deklarasi* (urutan
  kolom indikator dan aturan buang `len(var.values) < 2`, `continuize.py:21`). Nilai
  deklarasi yang tidak muncul di data tetap jadi kolom 0 (Klaim 7).
- Imputasi mean, lalu kolom seluruhnya NaN dibuang.
- Normalize: SD ddof=0, tanpa centering, SD 0/NaN → 1. Salah ddof tidak tertangkap test
  `components_`/rasio, hanya test skor dan `explained_variance_` (Klaim 3).

Keputusan ini juga jadi preseden untuk model Tahap 2 (metaheuristik), yang akan
membutuhkan preprocessing serupa.

## Opsi

1. **zestcore mengerjakan semua (Continuize, imputasi, Normalize).** Input: X mentah +
   metadata kolom dalam bentuk NumPy/Python biasa. zestcore bisa dipakai di luar Orange
   dengan hasil setara. Kode paling banyak; semua aturan di atas harus ditiru dan diuji.
2. **Pemanggil mengerjakan semua.** Lapis widget memakai preprocessor Orange sendiri;
   zestcore hanya menerima X numerik siap pakai (langkah 4-7). Kesamaan preprocessing
   dijamin karena kodenya kode Orange. zestcore saja tidak setara dengan PCA Orange
   kecuali penggunanya memproses data dengan cara yang sama.
3. **Dibagi: pemanggil Continuize, zestcore imputasi + Normalize.** Bagian yang butuh
   metadata variabel Orange tetap di Orange; langkah numerik ada di zestcore.

## Konsekuensi

- Opsi 1: klaim Tahap 1 ("PCA NumPy murni identik dengan Orange") berlaku dari data
  mentah. zestcore perlu representasi metadata kolom sendiri.
- Opsi 2: inti zestcore paling kecil; klaim "identik" berlaku dari X yang sudah
  diproses. Test harus memberi X hasil preprocessing Orange ke zestcore.
- Opsi 3: batasnya ada di tengah pipeline, jadi kedua lapis harus sepakat soal urutan
  kolom dan NaN; test perlu menguji sambungannya.
- Rumus jumlah arah varians nol (aturan test poin 2) butuh metadata nilai deklarasi;
  di opsi 2 dan 3 rumus itu dihitung di sisi pemanggil atau test.

## Keputusan

Saya memilih opsi 2: pemanggil mengerjakan semua preprocessing.

Lapis widget dan test memakai preprocessor Orange sendiri (Continuize → SklImpute →
Normalize). zestcore hanya menerima X numerik siap pakai dan mengerjakan langkah 4-7:
n_components, centering + kovarians ddof=1 + eigh, tanda `svd_flip`, dan skor. Dengan
begitu, kesamaan preprocessing dijamin karena memakai kode Orange, dan aturan
CLAUDE.md (zestcore NumPy murni, tanpa `import Orange`/`sklearn`) tetap terjaga tanpa
meniru metadata variabel Orange.

Klaim Tahap 1 ditulis sebagai: "PCA NumPy murni di zestcore identik dengan PCA Orange
untuk X yang sudah diproses preprocessor Orange." Test memberi X hasil preprocessing
Orange ke zestcore, dan rumus jumlah arah varians nol dihitung di sisi test. Untuk
model Tahap 2, ini jadi preseden: zestcore menerima data numerik siap pakai, kecuali
ada model yang memang butuh preprocessing khusus.
