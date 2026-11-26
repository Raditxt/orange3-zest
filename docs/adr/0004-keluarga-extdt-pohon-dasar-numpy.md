---
status: proposed
---

# Keluarga model ExtDT di atas pohon dasar NumPy

## Konteks

Arahan Prof. Irfan (23 Nov 2026): model pertama Tahap 2 adalah "extended decision tree"
(ExtDT). Kajian T2b menemukan bahwa istilah ini tidak menunjuk ke satu algoritma, tetapi
ke beberapa kandidat. Pembandingnya, Tree milik Orange 3.40.0, sudah dibedah di
`docs/orange-tree-internals.md` (Klaim 1-7). ADR ini memakai ADR 0002 (preprocessing
lewat preprocessor Orange) dan ADR 0003 (pola tiga lapis: zestcore, lalu adapter
Orange, lalu widget tipis; untuk ExtDT adapternya berupa pasangan `Learner`/`Model`).

Fakta yang mengikat semua opsi di bawah. Path relatif ke
`D:\Orange\Lib\site-packages\`.

- **Tree Orange bukan sklearn.** Widget Tree memakai `TreeLearner` Orange: information
  gain dalam bit, threshold = nilai data maksimum di sisi kiri, NaN tidak diimputasi,
  dan atribut diskret ditangani langsung (Klaim 1-5).
- **sklearn `ExtraTreesClassifier` (1.5.2).**
  - Threshold diambil acak seragam di [min, max] fitur pada node
    (`sklearn/tree/_splitter.pyx:785-792`).
  - Kriteria hanya `gini`, `entropy`, atau `log_loss` (`sklearn/tree/_classes.py:941`).
    Skor ternormalisasi 2I/(Hs+Hc) dari Geurts dkk. (2006) tidak tersedia. Kriteria
    baru hanya bisa ditambahkan sebagai subclass `Criterion` Cython, bukan NumPy.
  - NaN ditolak, karena `allow_nan` hanya berlaku untuk `splitter="best"`
    (`_classes.py:1085-1090`). Dicek: fit dengan satu NaN menghasilkan `ValueError:
    Input X contains NaN`.
- **Pola Orange untuk membungkus sklearn.** `RandomForestLearner` adalah `SklLearner`
  (`Orange/classification/random_forest.py:42-44`) dengan preprocessor bawaan
  `HasClass, Continuize, RemoveNaNColumns, SklImpute` (`Orange/base.py:549-553`).
  Atribut diskret di-one-hot dan NaN diimputasi sebelum sampai ke sklearn.
- **Preprocessor dari widget menggantikan preprocessor bawaan.** Aturan ini ada di
  `Orange/base.py:53-58`. Pengecualiannya learner `Fitter`, yang diberi
  `use_default_preprocessors = True` (`Orange/widgets/utils/owlearnerwidget.py:203-204`).
  Untuk learner non-Fitter, widget hanya menampilkan info "Ignoring default
  preprocessing" (`owlearnerwidget.py:87-91, 216-219`).
- **Temuan T2b-1 (Klaim 1).** Pada `Fitter`, kwarg `preprocessors` dibuang di jalur
  regresi, karena `__init__` regresi tidak punya nama itu (`Orange/modelling/base.py:82-86`).

## Keputusan

Raditya memutuskan (26 Nov 2026):

> ExtDT dikerjakan sebagai keluarga model di atas satu pohon dasar NumPy, dengan
> urutan: Extra Trees (pilot, Geurts 2006, pembanding sklearn `ExtraTreesClassifier`)
> → DT multi-label → C4.5 (pembanding Weka J48) → UDT → Extension DT (diawali riset
> lanjutan: cari dataset publik + angka acuan).

Lima pertanyaan di bawah **belum diputuskan**. Tiap pertanyaan memuat opsi beserta
konsekuensinya, dan pilihannya diserahkan kepada Raditya.

Catatan istilah: "ExtDT" sekarang dipakai untuk nama keluarga, sedangkan "Extension DT"
adalah salah satu anggotanya, dan "Extra Trees" adalah anggota lain. Ketiganya mudah
tertukar di laporan. Nama kanoniknya perlu ditetapkan sebelum kode pertama.

## Pertanyaan 1. Tulis dari nol (NumPy) atau bungkus sklearn

### 1a. Pohon dasar ditulis dari nol di zestcore (NumPy murni)

- Sesuai aturan CLAUDE.md dan ADR 0003: zestcore berisi algoritma, adapter hanya
  membungkus.
- Skor 2I/(Hs+Hc) bisa ditulis langsung, begitu juga gain ratio C4.5, entropi
  multi-label, dan bobot pecahan UDT. Semua anggota keluarga berbagi kode yang sama.
- Hasil pilot tidak akan identik dengan sklearn, karena generator bilangan acak dan
  urutan pengambilan atributnya berbeda. Yang bisa diklaim hanya kesetaraan distribusi
  (Pertanyaan 3).
- Risiko kecepatan: pembangunan node berjalan di Python. Extra Trees relatif ringan
  (satu threshold acak per atribut kandidat, bisa divektorkan per node), tetapi ensemble
  100 pohon pada bank-full (45211 baris) belum pernah diukur. Waktunya harus diukur,
  bukan diasumsikan.

### 1b. Bungkus sklearn `ExtraTreesClassifier` seperti `RandomForestLearner`

- Kodenya paling sedikit (pola `SklLearner`). Untuk seed yang sama, hasilnya identik
  dengan sklearn, dan cepat.
- Tidak ada kode di zestcore, sehingga aturan "orangecontrib hanya membungkus zestcore"
  dilanggar, atau perlu pengecualian tertulis.
- Skor paper 2I/(Hs+Hc) tidak bisa dipakai tanpa menulis `Criterion` Cython. Akibatnya
  pilot menjadi "Extra Trees versi sklearn", bukan versi Geurts 2006.
- Tidak ada bagian yang bisa dipakai ulang oleh C4.5, UDT, maupun multi-label. Keputusan
  "satu pohon dasar" baru mulai berlaku di anggota kedua, jadi pilot tidak menguji
  pohon dasar sama sekali.

### 1c. Pohon dasar NumPy dengan skor yang bisa dipilih: `entropy` dan `paper`

Sama dengan 1a, ditambah mode skor `entropy` (information gain biasa, seperti
`criterion="entropy"` di sklearn) di samping skor paper 2I/(Hs+Hc).

- Perbandingan dengan sklearn bisa dipisah menjadi dua langkah:
  1. Mode `entropy` vs sklearn: hanya berbeda di bilangan acak.
  2. Mode `paper` vs mode `entropy`: hanya berbeda di fungsi skor.
  Pada 1a, kedua perbedaan ini tercampur dalam satu angka.
- Biayanya satu fungsi skor tambahan beserta test-nya. Di widget, mode ini bisa
  disembunyikan.

## Pertanyaan 2. Batas pohon dasar dan titik perluasan

Tujuannya, E/D/C (anggota berikutnya) cukup *menambah* kode, bukan *mengubah* pohon
dasar. Empat titik perluasan dari brief (kriteria split, pemilih threshold, aturan stop,
agregasi ensemble) ternyata belum cukup. Tabel berikut memetakan kebutuhan tiap anggota.
Sumbernya pengetahuan umum tentang algoritma tersebut dan belum dicek ulang ke paper
masing-masing (lihat Status).

| Kebutuhan | Extra Trees | Multi-label | C4.5 | UDT | Extension DT |
|---|---|---|---|---|---|
| Kriteria split | 2I/(Hs+Hc) | entropi multi-label | gain ratio | entropi berbobot | ? |
| Pemilih threshold | acak di [min, max] node | exhaustive | nilai data (≤ titik tengah) | exhaustive atas titik pdf | ? |
| Pemilihan atribut | K acak | semua | semua | semua | ? |
| Bentuk target y | 1 kolom kelas | **matriks n × L** | 1 kolom | 1 kolom | ? |
| Bobot baris | tidak | tidak | **ya** (missing pecahan) | **ya** (tuple pecahan) | ? |
| Split diskret multiway | opsional | opsional | **ya** | ? | ? |
| Pasca-bangun | tidak | tidak | **pruning** | ? | ? |
| Agregasi ensemble | ya | tidak | tidak | tidak | ? |

Yang tidak tercakup empat titik dari brief ditandai tebal:

- **Bentuk target.** Multi-label mengubah daun dari "jumlah per kelas" menjadi "frekuensi
  per label". Kalau pohon dasar mengasumsikan y 1-D, anggota kedua sudah harus mengubah
  pohon dasar.
- **Bobot baris.** C4.5 dan UDT membagi satu baris ke beberapa cabang dengan bobot
  pecahan. Kalau bobot tidak ada sejak awal, anggota ketiga harus mengubah pohon dasar.
- **Pruning.** Ini langkah setelah pohon selesai dibangun, bukan bagian dari aturan
  stop.

### 2a. Fungsi strategi (parameter callable)

`build_tree(X, y, w, *, score, propose_splits, should_stop, rng)` di zestcore. Tiap anggota
keluarga berupa satu set fungsi. Ensemble dan pruning menjadi fungsi terpisah yang
memanggil atau menerima pohon.

- NumPy murni, dan tiap fungsi bisa diuji sendiri dengan angka brute-force, seperti di
  `tree_internals_check.py`.
- Konsistensi antarfungsi tidak dijaga oleh struktur. Misalnya, skor multi-label yang
  diberi y 1-D baru gagal saat dijalankan, jadi perlu validasi di awal.

### 2b. Kelas dasar dengan method yang di-override

`BaseTree` punya `_score`, `_propose_splits`, `_should_stop`, dan `_leaf_value`.
Tiap anggota keluarga menjadi subclass.

- Lebih mudah dibaca bagi pembaca yang terbiasa OOP, dan cocok dengan pasangan
  `Learner`/`Model` Orange.
- Ada risiko kopling warisan. Subclass sering ikut bergantung pada urutan pemanggilan
  internal, sehingga perubahan di `BaseTree` bisa merusak anggota lain tanpa terlihat.

### 2c. Kontrak data saja, tanpa kerangka bersama

Pohon dasar hanya menetapkan struktur node (array `feature`, `threshold`, `children`,
`value`, `weight`) dan fungsi prediksi. Tiap anggota menulis builder-nya sendiri.

- Paling sederhana untuk pilot, dan kontrak node tetap dipakai bersama.
- Logika rekursi dan aturan stop terduplikasi di tiap anggota. Ini menyimpang dari
  maksud keputusan "satu pohon dasar".

Konsekuensi yang berlaku untuk semua opsi: kalau `y` 2-D dan `w` tidak masuk kontrak
sejak pilot, janji "menambah, bukan mengubah" tidak berlaku untuk multi-label maupun
C4.5. Dua hal ini tidak menambah kode pilot secara berarti (y bisa `(n, 1)` dan w bisa
semua 1), tetapi harus diuji sejak awal.

## Pertanyaan 3. Klaim kebenaran pilot

Hasil pilot tidak bisa identik dengan sklearn (Pertanyaan 1). Sesuai CLAUDE.md, hal ini
adalah temuan yang sah dan penyebabnya harus ditulis. Ada tiga lapis klaim yang bisa
diambil sebagian atau semuanya.

### 3a. Komponen deterministik: identik

Ini mencakup fungsi skor vs perhitungan brute-force, threshold selalu berada di
[min, max] node, aturan stop, dan prediksi pohon yang dibangun manual. Hasilnya diuji
dengan `np.allclose` dan selisih maksimum dilaporkan, seperti Tahap 1. Klaim ini kuat
dan murah.

### 3b. Ensemble: setara distribusi dengan sklearn

- Setting disamakan: `n_estimators`, `max_features` = K, `min_samples_split` = n_min,
  `bootstrap=False`, `criterion="entropy"` (dibandingkan dengan mode `entropy` di 1c).
- Desain eksperimen: R seed × CV berlapis k-fold berulang (misalnya 10 × 10), lalu
  rata-rata ± sd akurasi/AUC per dataset.
- **Kriteria "setara" harus ditulis sebelum melihat angka.** Contohnya: selisih rata-rata
  < x, atau uji berpasangan per fold dengan ambang tertentu. Kalau kriterianya
  ditetapkan setelah melihat angka, itu sama saja dengan menutupi selisih memakai
  toleransi longgar.
- Perbedaan yang sudah diketahui dan harus dicatat:
  - RNG dan urutan pengambilan atribut berbeda.
  - sklearn melewati atribut konstan saat mengambil K atribut.
  - Agregasi berbeda. sklearn merata-ratakan probabilitas, sedangkan Geurts 2006
    (sejauh yang saya ingat) memakai voting mayoritas untuk klasifikasi. Ini perlu
    dicek di paper.

### 3c. Performa vs Tree dan Random Forest Orange

Perbandingan lewat Test and Score dengan fold yang sama, plus bukti `.ows`. Ini bukan
klaim kebenaran, melainkan klaim kegunaan. Setting pembanding harus disebut:

- Tree memakai default widget (`binarize=True`, dst., Klaim 1).
- RF memakai Continuize + SklImpute.

Konsekuensi: 3a + 3b membuktikan "implementasi benar", 3c menjawab "apakah berguna".
Tanpa 3b, klaim pilot hanya "masuk akal". Tanpa 3a, kegagalan di 3b sulit dilacak
penyebabnya.

## Pertanyaan 4. Missing value dan atribut diskret: di pohon atau di preprocessing

ADR 0002 menetapkan preprocessing lewat preprocessor Orange dan zestcore hanya menerima
X numerik siap pakai, "kecuali ada model yang memang butuh preprocessing khusus". Tree
Orange sendiri berada di pengecualian itu, karena tidak punya preprocessor (Klaim 1, 3, 5).

### 4a. Ditangani di pohon

zestcore menerima X dengan NaN, ditambah metadata sederhana berupa array NumPy biasa:
`is_discrete` dan `n_values` per kolom.

- Ini satu-satunya cara agar C4.5 bisa dibandingkan dengan J48, yang punya split diskret
  multiway dan missing pecahan secara internal. UDT juga butuh representasi khusus.
- Ini menjadi pengecualian ADR 0002 yang pertama, dan harus ditulis. zestcore mulai
  membawa metadata kolom, yang dulu dihindari di ADR 0002 opsi 1.
- Adapter tetap memakai `HasClass` (baris dengan target NaN dibuang). Ini sama dengan
  Orange, Klaim 5a.

### 4b. Preprocessing Orange (pola `SklLearner`: Continuize + Impute)

- Konsisten penuh dengan ADR 0002, dan zestcore tetap paling kecil.
- Pembandingan pilot dengan sklearn menjadi adil, karena sklearn juga menerima data
  one-hot dan sudah diimputasi.
- One-hot mengubah perilaku Extra Trees. K atribut acak diambil dari kolom indikator,
  bukan dari atribut asli, sehingga atribut diskret dengan banyak nilai lebih sering
  terpilih. Ini menyimpang dari Geurts, yang menangani atribut kategorikal secara
  langsung.
- C4.5 dan UDT nanti tetap butuh 4a, jadi kontrak input zestcore berubah di tengah jalan.

### 4c. Bertahap: pilot memakai 4b, kontrak input sudah memuat slot untuk 4a

`build_tree` sejak awal menerima `is_discrete`/`n_values` (pilot: semua kontinu) dan
mengizinkan NaN di kontrak. Pilot sendiri menolak NaN dengan pesan yang jelas.

- Pilot tetap sederhana, dan anggota berikutnya cukup menambah tanpa mengubah tanda
  tangan fungsi.
- Ada parameter yang belum dipakai di pilot, dan ini harus dijelaskan saat
  dipertanggungjawabkan.

Konsekuensi untuk semua opsi: kalau adapter punya preprocessor bawaan dan pengguna
menyambungkan widget Preprocess, preprocessor bawaan **diganti**, bukan ditambah
(`Orange/base.py:53-58`). Pada opsi 4b, ini berarti NaN bisa sampai ke zestcore lewat
GUI. Validasi input di adapter atau di zestcore wajib ada di semua opsi.

## Pertanyaan 5. Preprocessor yang hilang di jalur regresi

Temuan T2b-1: `Fitter` membuang kwarg `preprocessors` bila `__init__` learner regresi
tidak menyebut nama itu. Untuk ExtDT, artinya preprocessor dari widget hilang tanpa
pesan apa pun bila adapter memakai `Fitter` dan learner regresinya lalai menyebut
parameter itu.

### 5a. Pilot klasifikasi saja, tanpa `Fitter`

`ZestExtraTreesLearner` langsung berupa `Learner` klasifikasi.

- Masalah ini tidak muncul di pilot. Extra Trees memang punya varian regresi, tetapi
  sklearn pembandingnya klasifikasi.
- Masalahnya hanya ditunda. Begitu regresi ditambahkan, Pertanyaan 5 muncul lagi.
  Learner non-Fitter juga tidak mendapat `use_default_preprocessors = True` dari widget
  (Pertanyaan 4).

### 5b. `Fitter` sendiri, kedua learner wajib menerima `preprocessors` + test penanda

Kedua `__init__` menyebut `preprocessors` secara eksplisit. Test meniru
`check_klaim1`: penanda harus sampai ke learner klasifikasi **dan** regresi.

- Satu widget bisa melayani dua jenis target, seperti widget Tree, dan default
  preprocessor tetap aktif karena Fitter (`owlearnerwidget.py:203-204`).
- Kodenya lebih banyak, dan test penanda harus dijaga untuk setiap anggota keluarga.

### 5c. Tanpa input Preprocessor dari widget

Preprocessor ditetapkan di kelas learner, dan widget tidak meneruskan input
Preprocessor.

- Masalahnya hilang sepenuhnya dan perilakunya mudah dijelaskan.
- Berbeda dengan semua widget learner Orange, yang punya input Preprocessor
  (`owlearnerwidget.py:92`). Pengguna yang terbiasa menyambungkan Preprocess akan
  heran saat inputnya tidak ada.

Konsekuensi untuk semua opsi: bentuk adapter yang dipilih menjadi template untuk semua
anggota keluarga, jadi pilihannya sebaiknya diambil sebelum pilot. Test penanda (5b)
bisa ditambahkan di opsi apa pun dan murah.

## Status

### Handoff 26 Nov 2026 (T2b-2)

**Selesai**

- Draf ADR ini (status `proposed`), dengan keputusan urutan keluarga dari Raditya dan
  lima pertanyaan beserta opsinya. Tidak ada kode zestcore.
- Fakta sklearn 1.5.2 dan Orange 3.40.0 di bagian Konteks dicek ke kode yang
  terpasang. Fakta NaN di `ExtraTreesClassifier` dicek dengan menjalankan fit.
- Notion: Catatan task T2b diisi dan status diubah ke Selesai. 📊 Laporan dikirim ke PM.

**Belum terverifikasi / perlu diperhatikan**

- Klaim tentang paper di ADR ini awalnya ditulis dari pengetahuan umum, karena file riset
  belum ada saat draf dibuat. Pencocokan dengan `docs/research/extdt-kandidat-rujukan.md`
  (commit e312660) ada di bagian "Pencocokan klaim paper" di bawah. Opsi dan rekomendasi
  tidak diubah.
- Nama kanonik: ExtDT, Extension DT, Extra Trees (lihat catatan istilah). Belum ada
  `GLOSSARY.md`. Dibuat setelah nama ditetapkan.

### Pencocokan klaim paper dengan file riset (26 Nov 2026)

Arti label:

- **cocok**: isinya sama dengan file riset, dan file riset tidak menandainya
  [belum terverifikasi].
- **berbeda**: file riset menyatakan hal lain.
- **belum terverifikasi**: file riset tidak membahasnya, atau menandainya
  [belum terverifikasi].

File riset sendiri bukan sumber primer, jadi label "cocok" berarti sesuai dengan file
riset, bukan sudah dicek langsung ke PDF.

| No | Klaim di ADR 0004 (lokasi) | Isi file riset | Status |
|---|---|---|---|
| 1 | Skor Extra Trees di Geurts 2006 = 2I/(Hs+Hc), tidak ada di sklearn (Konteks, 1b, 1c, tabel P2) | Bagian A: paper memakai normalized information gain 2·I/(H_s+H_c), sklearn gini/entropy/log_loss | **cocok** |
| 2 | Geurts 2006 memakai voting mayoritas, sklearn rata-rata probabilitas (3b) | Tidak dibahas sama sekali | **belum terverifikasi**: tetap harus dicek ke PDF Geurts |
| 3 | Threshold C4.5 = nilai data terbesar ≤ titik tengah (tabel P2) | Bagian D hanya menyebut gain ratio, missing fraksional, pruning CF=0,25, subtree raising, dan koreksi MDL. Aturan threshold tidak disebut | **belum terverifikasi**. Yang **cocok** dengan tabel P2: gain ratio, bobot untuk missing pecahan, dan pruning |
| 4 | UDT butuh bobot baris (tuple pecahan), entropi berbobot, threshold exhaustive atas titik pdf (tabel P2) | Bagian C: UDT mencari split atas interval pdf (s=100 titik sampel). Ekspansi `sample_weight` disebut "hanya aproksimasi, bukan pencarian split ala UDT (inferensi)". Tuple pecahan tidak disebut | **berbeda sebagian**: pencarian split atas pdf cocok, tetapi file riset menyatakan bobot baris saja tidak cukup untuk UDT. Tuple pecahan **belum terverifikasi** |
| 5 | Multi-label: entropi multi-label (Clare & King 2001), target n × L, daun = frekuensi per label (tabel P2, P2 poin "Bentuk target") | Bagian E: entropi −Σ[p log p + q log q] dan daun berisi **himpunan label**. Gain vs gain ratio dan aturan pelabelan daun **[belum terverifikasi]** | Entropi **cocok**. Isi daun **berbeda**: paper menyebut himpunan label, ADR menyebut frekuensi per label. Aturan pelabelan daun **belum terverifikasi** |

Temuan lain dari file riset yang menyentuh teks ADR. Ini dicatat saja, teks opsinya
tidak diubah.

- **1b** menulis bahwa membungkus sklearn tidak memberi bagian yang bisa dipakai ulang
  oleh multi-label. Menurut bagian E, `DecisionTreeClassifier` dengan y 2-D dan
  `criterion="entropy"` memberi impurity yang setara dengan entropi ML-C4.5 dibagi L.
  Artinya sklearn tetap bisa jadi pembanding parsial untuk multi-label.
  - Uji 298/300 di file riset dijalankan dengan sklearn 1.9.1, bukan 1.5.2.
  - Uji itu juga belum dipublikasikan.
- **Tabel P2, Extra Trees, "split diskret multiway: opsional".** Menurut bagian A, paper
  memakai split subset acak (dua cabang) untuk atribut nominal, bukan multiway.
- **Tabel P2, Extension DT, semua "?".** Bagian B: model matter-element, diskretisasi dengan
  dependent function, lalu pohon ID3 (information gain), lalu extension transform pada
  aturan. Tidak ada kode, dataset publik, atau angka akurasi.
- **Pembulatan K.** sklearn memakai floor, paper "rounded" (bagian A, ditandai
  [belum terverifikasi]). Ini relevan untuk 3b, karena `max_features` harus disamakan.
- **Multi-target di adapter.** Bagian E: `TreeLearner`/`SklTreeLearner` tidak
  menyetel `supports_multiclass`, sehingga data multi-target memicu `TypeError`. Adapter
  multi-label butuh `supports_multiclass = True`.

**Langkah berikutnya (urutan usulan, jadwal diatur PM)**

1. Raditya memilih opsi untuk Pertanyaan 1-5, lalu menulisnya di bagian Keputusan dan
   mengubah status menjadi `accepted`.
2. Cek ke PDF klaim yang masih belum terverifikasi: agregasi Geurts (no. 2), threshold
   C4.5 (no. 3), tuple pecahan UDT (no. 4), dan isi daun ML-C4.5 (no. 5).
3. Tulis kriteria "setara" (3b) sebelum eksperimen.
4. Pecah pilot menjadi task implementasi: contoh kecil fungsi skor di iris (3a), lalu
   pohon tunggal, ensemble, adapter, dan widget.
