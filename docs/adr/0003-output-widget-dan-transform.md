---
status: accepted
---

# Output widget Zest PCA dan transform untuk data baru

## Konteks

Task T2a (arahan Prof. Irfan, 23 Nov 2026): widget Zest PCA membungkus `zestcore.pca`,
mengeluarkan minimal Transformed Data dan Components, dan bisa memproyeksikan data baru.
Widget ini jadi template untuk model berikutnya (ExtDT). Preprocessing memakai
preprocessor Orange (ADR 0002).

Cara OWPCA melakukannya (Orange 3.40.0, path relatif ke `Orange/`):

- Domain Transformed Data dibangun di `DomainProjection.__init__`
  (`projection/base.py:190-211`). Tiap komponen adalah `ContinuousVariable` dengan
  `compute_value=ComputeValueProjector(feature=i, transform=TransformDomain(self))`.
- `TransformDomain.__call__` mengubah data ke `pre_domain` (domain hasil
  preprocessing, disimpan di `projection/base.py:65,68`), lalu memanggil
  `proj.transform(X)` (`projection/base.py:139-142`, `115-116`).
  `ComputeValueProjector.compute` hanya mengambil kolom ke-i (`projection/base.py:175-176`).
  Hasil `TransformDomain` di-cache per tabel sumber, jadi proyeksi dihitung sekali untuk
  semua komponen (`data/table.py:312-317`).
- Apply Domain hanya menjalankan `data.transform(template.domain)`
  (`widgets/data/owtransform.py:27`). Variabel `pre_domain` membawa compute_value
  preprocessor dengan statistik data latih, misalnya Normalize menyimpan SD latih
  (`preprocess/normalize.py:47-67`). Data baru otomatis diproses dengan statistik latih.
- Skor data latih juga lewat jalur ini: `self._pca(self.data)` (`owpca.py:321`) memanggil
  `data.transform(self.domain)` (`projection/base.py:118-119`). Skor tidak diambil dari
  hasil fit.
- Output Components sengaja tanpa transform: `compute_value=lambda _: None`
  (`owpca.py:336-355`).
- Variabel tanpa compute_value yang tidak ada di tabel sumber diisi nilai kosong
  (`data/table.py:307-310`).

`zestcore.pca` (`zestcore/pca.py:17`) sekarang hanya mengembalikan skor untuk X yang
dipakai fit. Belum ada fungsi untuk memproyeksikan X baru.

## Opsi

Opsi 1a-1c sama-sama menghasilkan Transformed Data yang bisa dipakai Apply Domain.
Perbedaannya ada di siapa yang membangun domain dan menjalankan preprocessing.

### 1a. Adaptor diserahkan ke `DomainProjection` Orange

Kelas adaptor di `orangecontrib/zest/` menyimpan `components_` dan `mean_`, dengan
`transform(X)` yang memanggil zestcore. Widget menjalankan preprocessor sendiri, lalu
membuat `DomainProjection(adaptor, pre_domain, k)` dan mengisi `pre_domain` serta `name`
secara manual. Kedua atribut itu biasanya diisi `Projector.__call__`
(`projection/base.py:65-69`).

- Jumlah kode: paling sedikit, sekitar 15-20 baris adaptor plus preprocessing di widget.
- Ketergantungan: `DomainProjection` ada di `__all__` (`projection/base.py:21-22`),
  tetapi kontraknya implisit. Adaptor harus punya atribut yang dibaca lewat
  `__dict__.update(proj.__dict__)` (`projection/base.py:112`, `201-202`). `pre_domain`
  diisi dari luar, padahal di Orange atribut itu diisi `Projector`.
- Skor latih lewat compute_value: ya, bila widget memanggil `model(data)` seperti
  `owpca.py:321`.
- Pickle: Transformed Data yang disimpan sebagai `.pkl` membawa adaptor di
  compute_value. Kelas adaptor harus didefinisikan di tingkat modul, dan file hanya bisa
  dibuka bila orange3-zest terpasang. Save Model tidak relevan karena inputnya
  `Orange.base.Model` (`widgets/model/owsavemodel.py:22`), sedangkan proyeksi bukan
  `Model`. Hal ini berlaku untuk semua opsi.

### 1b. compute_value sendiri dengan pola `TransformDomain`/`ComputeValueProjector`

`orangecontrib/zest/` menulis kelas compute_value bersama (subclass `SharedComputeValue`)
dan per kolom, lalu membangun `Domain` sendiri. `TransformDomain` di sini juga
mengerjakan konversi ke `pre_domain`.

- Jumlah kode: paling banyak, sekitar 50-70 baris. Termasuk `__eq__`/`__hash__`,
  yang wajib supaya cache tabel benar (`data/util.py:77-84`, `projection/base.py:144-162`).
- Ketergantungan: hanya `SharedComputeValue`, `Domain`, dan `ContinuousVariable`, yang
  merupakan API data publik. Tidak bergantung pada `Orange.projection`.
- Skor latih lewat compute_value: ya, bila widget memanggil `data.transform(domain)`.
- Pickle: sama dengan 1a. Kelas compute_value harus di tingkat modul dan perlu
  orange3-zest terpasang. Fungsi lokal atau lambda tidak bisa di-pickle.

### 1c. `ZestPCA(Projector)` + `ZestPCAModel(DomainProjection)`

Pasangan ini meniru `PCA`/`PCAModel` (`projection/pca.py:27-53`, `73-78`) dan
`LinearProjector` (`projection/base.py:241-259`). `ZestPCA.preprocessors` berisi
Continuize → SklImpute (→ Normalize bila dicentang). `fit(X)` memanggil `zestcore.pca`
dan mengembalikan `ZestPCAModel`. `Projector.__call__` menjalankan preprocessing dan
mengisi `pre_domain` (`projection/base.py:54-71`).

- Jumlah kode: sedikit, sekitar 30-40 baris, dan widget tetap tipis.
- Ketergantungan: `Projector` dan `DomainProjection` (keduanya di `__all__`), dipakai
  persis sesuai pola subclass Orange sendiri. Kontrak `proj.transform`/`components_`
  dan `mean_` tetap implisit seperti di 1a.
- Skor latih lewat compute_value: ya, jalurnya sama dengan OWPCA (`model(data)`).
- Pickle: sama dengan 1a. `Projector` sudah menangani state thread-local
  (`projection/base.py:99-106`). `ZestPCA` juga bisa dikirim sebagai output setara
  output `PCA` milik OWPCA (`owpca.py:364`).

### 2. Tabel NumPy biasa tanpa transform

Widget memanggil `zestcore.pca`, lalu membuat `Table` dari `result.scores` dengan
variabel PC tanpa compute_value.

- Jumlah kode: sedikit, sekitar 15 baris di widget, dan tidak butuh fungsi transform di
  zestcore.
- Ketergantungan: hanya `Table`, `Domain`, dan `ContinuousVariable`.
- Skor latih lewat compute_value: tidak. Skor diambil langsung dari `result.scores`,
  jadi jalurnya beda dengan OWPCA.
- Pickle: `.pkl` bisa dibuka tanpa orange3-zest. Data baru tidak bisa diproyeksikan:
  Apply Domain menghasilkan kolom PC kosong (`data/table.py:307-310`). Syarat T2a
  ("transform untuk data baru") tidak terpenuhi.

## Konsekuensi

- **zestcore butuh `transform(X, mean, components)` untuk opsi 1a-1c.** Rumusnya harus
  ditaruh di zestcore. Kalau ditulis di lapis widget, aturan "hanya membungkus" di
  CLAUDE.md terlanggar. Bentuk rumus menentukan seberapa dekat hasilnya dengan sklearn:
  - sklearn menghitung `X @ C.T - mean @ C.T` (`sklearn/decomposition/_base.py:148-155`).
  - `(X - mean) @ C.T` setara secara matematis, tetapi beda urutan floating-point. Hasilnya
    hanya identik dalam toleransi spec, tidak bit-identik.
  - `zestcore.pca` sudah memakai bentuk sklearn untuk skor latih (`zestcore/pca.py:64-65`).
    `transform` sebaiknya memakai bentuk yang sama, dan `pca` memanggilnya, supaya hanya
    ada satu sumber rumus.
  - Apakah hasilnya bit-identik dengan Orange tetap harus diukur (selisih maksimum
    dilaporkan), karena `mean` dan `components` sendiri berasal dari jalur hitung zestcore.
- **Template untuk ExtDT: pola tiga lapis.**
  1. `zestcore`: algoritma NumPy murni.
  2. Kelas adaptor Orange di `orangecontrib/zest/`: `Projector`/`DomainProjection` untuk
     proyeksi, `Learner`/`Model` untuk ExtDT.
  3. Widget tipis yang hanya mengatur setting, memanggil adaptor, dan mengirim output.

  Untuk learner, preprocessing dan penyimpanan `original_domain` dikerjakan
  `Learner.__call__` (`Orange/base.py:124-162`). Data baru diubah ke domain model oleh
  `Model.data_to_model_domain` (`Orange/base.py:343-386`) dan masuk lewat widget
  Predictions. Mekanisme compute_value per kolom (`DomainProjection`,
  `TransformDomain`, `ComputeValueProjector`) tidak terpakai di ExtDT. Yang terbawa ke
  ExtDT adalah pola tiga lapis, preprocessing lewat preprocessor Orange (ADR 0002), dan
  kerangka widget. Dari keempat opsi, 1c yang bentuknya paling sejajar dengan pasangan
  `Learner`/`Model`. Pola pasangan seperti itu tidak ada di 1a, 1b, maupun 2.
- **Bukti identik di test dan `.ows`:** pada opsi 1a-1c, skor dibandingkan lewat jalur yang
  sama dengan OWPCA (`model(data)`). Pada opsi 2, test membandingkan `result.scores`
  dengan skor OWPCA dari jalur compute_value.

## Keputusan

Saya memilih opsi 1c: ZestPCA(Projector) dan ZestPCAModel(DomainProjection) di
orangecontrib/zest.

- fit() memanggil zestcore.pca pada X yang sudah melewati preprocessor Projector
  (Continuize → SklImpute → Normalize bila dicentang). zestcore tetap NumPy murni.
- zestcore mendapat fungsi transform(X, mean, components) dengan rumus yang sama dengan
  zestcore.pca dan sklearn: X @ C.T − mean @ C.T.
- Skor Transformed Data dihitung lewat compute_value, baik untuk data latih maupun data
  baru lewat Apply Domain, sama seperti OWPCA (owpca.py:321).
- Semua kelas yang tersimpan di domain hasil (model dan objek proj) didefinisikan di
  level modul supaya bisa di-pickle, dan ini diuji. Bentuk objek proj (ZestPCA sendiri
  seperti LinearProjector, atau objek kecil terpisah) diputuskan saat implementasi
  dengan contoh kecil.
- Klaim: hasil Zest PCA identik dengan PCA Orange dalam toleransi spec. Bit-identik
  hanya dicatat sebagai observasi.

Alasan: jalurnya sama dengan OWPCA, jadi perbandingannya setara. Preprocessing dan
statistik data latih ikut terbawa ke data baru tanpa kode tambahan (ADR 0002). Pola tiga
lapis (zestcore → kelas adaptor Orange → widget tipis) menjadi template untuk ExtDT,
dengan pasangan Learner/Model menggantikan Projector/DomainProjection.

## Catatan implementasi

Bentuk objek proj (25 Nov 2026, T2a langkah 4a): **objek kecil terpisah** `ZestPCAFit`,
dibuat baru di setiap `ZestPCA.fit()`. Objek ini membawa `mean_`, `components_`,
`explained_variance_`, dan `explained_variance_ratio_`. Rasio dibutuhkan widget untuk
`var.attributes["variance"]` (`owpca.py:325-328`). `transform(X)` memanggil
`zestcore.transform`. Polanya sama dengan PCA Orange, yang membungkus objek sklearn baru
(`projection/pca.py:51-53`).

Alternatifnya, `ZestPCA` sendiri sebagai proj (pola `LinearProjector`,
`projection/base.py:251-259`), ditolak berdasarkan contoh kecil pada iris
(`docs/proofs/proj_forms_check.py`):

- Model lama berubah saat projector yang sama di-fit ulang. `DomainProjection` menyalin
  `proj.__dict__` (`projection/base.py:112`), tetapi `transform` tetap memanggil
  `self.proj.transform` (`:115-116`). Setelah projector di-fit ulang pada `iris[:50]`,
  `model(iris)` dari fit pertama berubah dengan selisih maks 5.46, dan
  `model.components_` tidak lagi sama dengan `model.proj.components_`.
- Pickle domain hasil maupun model gagal (`cannot pickle '_thread._local'`). Salinan
  `__dict__` itu ikut membawa `_Projector__tls` ke model, sedangkan atribut ini hanya
  dibuang oleh `Projector.__getstate__` (`:99-102`), bukan oleh `DomainProjection`.
  Dengan objek terpisah, domain hasil dan model iris bisa di-pickle (2807 byte).

Aturan template: **`transform(X)` hanya memakai statistik hasil fit, tidak pernah
statistik dari X yang sedang ditransformasi.** `Table.transform`/`from_table` menjalankan
compute_value per potongan maksimal 5000 baris (`_FromTableConversion.max_rows_at_once`,
`data/table.py:385`, `423-448`). `compute_shared` (di sini `TransformDomain`) menerima
potongan baris itu (`data/table.py:303, 316`), bukan seluruh tabel, dan cache dikosongkan
setelah setiap potongan (`:446-448`). Bukti: `model(table)` pada bank-full (45211 baris)
memanggil `ZestPCAFit.transform` 10 kali (9 × 5000 + 211 baris), sedangkan iris cukup
sekali. Uji sensitivitas 4a menunjukkan akibatnya: bila `transform` memakai
`X.mean(axis=0)`, skor data latih bank-full ikut salah, walaupun tabelnya sama dengan
data fit, karena setiap potongan memakai mean-nya sendiri. Iris dan heart_disease tetap
lolos karena hanya satu potongan. Aturan ini juga berlaku untuk model berikutnya (ExtDT)
dan untuk preprocessor buatan sendiri.
