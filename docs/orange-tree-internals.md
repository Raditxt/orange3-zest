# Spesifikasi: cara Orange 3.40.0 membangun Tree (pembanding ExtDT)

**Status: DRAF untuk review Raditya (26 Nov 2026), task T2b-1.** Belum ada kode
zestcore untuk tree; dokumen ini hanya membedah pembandingnya.

Lingkungan: `D:\Orange\python.exe` (Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2,
NumPy 2.1.3). Semua `file:baris` merujuk ke kode yang terpasang di
`D:\Orange\Lib\site-packages\Orange\`, bukan GitHub master. Skrip bukti:
`docs/proofs/tree_internals_check.py` (satu fungsi `check_klaimN()` per klaim, pakai
`assert`, exit 0 = semua klaim terbukti). Data: iris (150 baris, 4 numerik, 3 kelas) dan
heart_disease (303 baris, 13 atribut campuran, NaN di `major vessels colored` (4) dan
`thal` (2)). Untuk regresi dipakai heart_disease dengan target `max HR`.

Kecuali disebut lain, semua angka memakai setting **default widget Tree**:
`binarize=True, min_samples_leaf=2, min_samples_split=5, max_depth=100,
sufficient_majority=0.95`.

Catatan: scorer inti (`Orange/classification/_tree_scorers.*.pyd`) adalah Cython
terkompilasi dan sumber `.pyx`-nya tidak ikut terpasang. Klaim tentang isinya (rumus
skor, kandidat threshold) dibuktikan secara perilaku: hasilnya dicocokkan dengan
perhitungan ulang brute-force NumPy di skrip bukti, bukan dengan membaca kode.

## Klaim

### Klaim 1. Widget Tree memakai TreeLearner milik Orange, bukan sklearn

- `widgets/model/owtree.py:62` `LEARNER = TreeLearner` dari `Orange.modelling.tree`,
  yaitu sebuah `Fitter` (`modelling/tree.py:22-28`) yang memilih learner berdasarkan
  tipe target (`modelling/base.py:69-80`): target diskret ke
  `Orange.classification.tree.TreeLearner`, target numerik ke
  `Orange.regression.tree.TreeLearner`. `SklTreeLearner` (bungkus
  `sklearn.tree.DecisionTreeClassifier`, `classification/tree.py:234-247`) **tidak**
  dipakai widget.
- Model keluarannya `Orange.tree.TreeModel` untuk keduanya.
- Pemetaan setting widget ke parameter learner (`owtree.py:107-117`):

  | Setting widget (default) | Parameter learner | Bila checkbox mati |
  |---|---|---|
  | Induce binary tree (ON) | `binarize=True` | `False` |
  | Min. instances in leaves (2) | `min_samples_leaf=2` | `1` |
  | Do not split subsets smaller than (5) | `min_samples_split=5` | `2` |
  | Limit maximal tree depth (100) | `max_depth=100` | `None` |
  | Stop when majority reaches (95%) | `sufficient_majority=0.95` | `1` |

  Perhatikan: default `binarize` di learner `False` (`classification/tree.py:63`), tapi
  widget mengirim `True`. Pembanding ExtDT harus menyebut setting mana yang dipakai.
- Learner tree tidak punya preprocessor bawaan (`preprocessors = ()` untuk keduanya):
  tidak ada imputasi, continuize, atau normalisasi. Atribut diskret dan NaN ditangani
  langsung oleh algoritma (Klaim 3, 5).
- **Temuan:** preprocessor dari sinyal input widget hanya sampai ke learner klasifikasi.
  `Fitter.__kwargs` menyaring kwargs dengan `__init__.__code__.co_varnames`
  (`modelling/base.py:82-86`), dan `__init__` regresi (`regression/tree.py:57-60`)
  tidak punya nama `preprocessors`, jadi kwarg itu dibuang dan `sufficient_majority`
  juga. Bukti: penanda `["PENANDA"]` muncul di learner klasifikasi, tidak di regresi.
  Dampaknya ke ExtDT: kalau pembanding regresi memakai Preprocess widget, jalur
  regresi Orange mengabaikannya. Belum diuji di GUI.

### Klaim 2. Kriteria split: information gain (klasifikasi), penurunan MSE (regresi)

- Klasifikasi: **information gain dengan entropi basis 2** (bit), bukan Gini dan bukan
  gain ratio. Untuk atribut multiway rumusnya ditulis langsung di
  `classification/tree.py:114-117`; untuk numerik dan binarisasi lewat
  `find_threshold_entropy` / `find_binarization_entropy` (Cython).
- Regresi: **penurunan MSE** = var(y) − Σ (nᵢ/n)·var(yᵢ), var dengan ddof=0
  (`find_threshold_MSE`, `find_binarization_MSE`, `compute_grouped_MSE`).
- Pemilihan atribut: skor terbaik tiap atribut dibandingkan dengan `sc > best_score`
  (`classification/tree.py:178`, `regression/tree.py:138`), jadi **bila seri, atribut
  dengan indeks terkecil menang**. Atribut dengan skor 0 ditolak.
- Bukti (brute-force NumPy atas semua kandidat split, lalu dicocokkan dengan akar
  pohon Orange):

  | Data | Akar Orange | Akar brute-force | Skor |
  |---|---|---|---|
  | iris | petal length | petal length | 0.9182958341 bit |
  | heart_disease | thal | thal | 0.2063531160 bit |
  | heart_disease, target max HR | slope peak exc ST | slope peak exc ST | 104.1088319516 |

  Di iris, `petal length` dan `petal width` seri (0.918 bit); yang dipilih
  `petal length` karena indeksnya lebih kecil. Skor Cython vs rumus NumPy:
  iris `sepal length` selisih 2.2e-16; regresi `age` selisih 1.1e-12.

### Klaim 3. Atribut diskret: multiway atau 2-grup optimal

- `binarize=False`: satu cabang per **nilai deklarasi** (`DiscreteNode`,
  `classification/tree.py:87-124`). heart_disease: akar `thal` dengan 3 cabang.
  Nilai yang jumlahnya < `min_samples_leaf` dikosongkan; cabangnya jadi `None`
  (`classification/tree.py:101-106, 190-191`).
- `binarize=True` dan atribut > 2 nilai: **pencarian exhaustive** semua 2^(k−1) − 1
  pengelompokan nilai ke dua cabang, urut kode Gray (`find_binarization_entropy`,
  `find_binarization_MSE`), dengan batas `min_samples_leaf` per grup. Hasilnya
  `MappedDiscreteNode` dengan `mapping` nilai→cabang. heart_disease: `thal` →
  mapping `[1, 0, 0]` (normal ke kanan, sisanya ke kiri), sama dengan brute-force.
- Atribut 2 nilai tetap `DiscreteNode` walau `binarize=True`
  (`classification/tree.py:129-130`, `regression/tree.py:95-96`). Di pohon
  heart_disease: semua atribut 2 nilai jadi `DiscreteNode`, 3 dan 4 nilai jadi
  `MappedDiscreteNode`.
- Batas `MAX_BINARIZATION = 16`: atribut diskret dengan > 16 nilai dan
  `binarize=True` menghasilkan `ValueError` di `fit_storage`
  (`classification/tree.py:205-212`). Dengan `binarize=False` data yang sama jalan.
- Tidak ada one-hot. Ini beda besar dengan sklearn, yang butuh atribut diskret
  di-encode dulu.

### Klaim 4. Threshold numerik = nilai data, bukan titik tengah

- Kandidat threshold adalah nilai unik yang ada di data; threshold yang dipilih
  **sama dengan nilai maksimum di sisi kiri**. Aturan cabang
  (`tree.py:119-121`): x ≤ t ke cabang 0, x > t ke cabang 1. Tiap sisi minimal
  `min_samples_leaf` baris.
- Bukti: 106 `NumericNode` di pohon iris, heart_disease (klasifikasi) dan heart_disease
  (regresi), semuanya memenuhi `threshold ∈ data` dan `threshold == max(sisi kiri)`.
- Pembanding sklearn (`criterion="entropy"`): akar iris membelah fitur yang sama dengan
  partisi yang sama, tapi threshold-nya **2.45 = (1.9 + 3.0)/2** (titik tengah), sedangkan
  Orange **1.9**. Prediksi data latih sama, prediksi data baru di antara 1.9 dan 3.0
  bisa berbeda. ExtDT yang ingin identik dengan Orange harus memakai konvensi
  "nilai kiri".
- Skor numerik dikalikan fraksi baris yang tidak NaN (`classification/tree.py:157`,
  `regression/tree.py:117`), lihat Klaim 5.

### Klaim 5. Missing value

- (a) Baris dengan target NaN dibuang sebelum membangun pohon
  (`classification/tree.py:214`, `regression/tree.py:173`). Iris dengan 3 target NaN:
  `root.subset` berisi 147 baris.
- (b) Tidak ada imputasi. Saat split, baris dengan NaN di atribut split diberi cabang −1
  (`classification/tree.py:119-120, 158-160`; `MappedDiscreteNode.branches_from_mapping`
  `tree.py:89-91`) sehingga **tidak masuk anak mana pun**; baris itu tetap tercatat di
  `subset` node induknya. heart_disease, akar `thal`: 303 baris, 2 NaN, anak menampung
  301.
- (c) Skor atribut dikalikan fraksi baris yang diketahui (numerik `tree.py:157`,
  diskret binarisasi `tree.py:141`, multiway `tree.py:118`; regresi di dalam scorer
  Cython). Bukti: 90 dari 150 nilai `petal length` iris dibuat NaN → skornya turun ke
  0.4 × 0.918, dan akar pindah ke `petal width`.
- (d) Prediksi: bila nilai atribut split NaN, penelusuran **berhenti di node itu** dan
  hasilnya distribusi node internal tersebut (`tree.py:162-170, 180-183`). Iris baris 0
  dengan `petal length` NaN → probabilitas [1/3, 1/3, 1/3] = distribusi akar
  [50, 50, 50]. Hal yang sama terjadi bila cabang tujuan `None`.

### Klaim 6. Parameter stop

Urutan cek di `_build_tree` (`classification/tree.py:183-203`,
`regression/tree.py:142-162`), untuk node di level `level` (akar = 1):

1. `len(node) < min_samples_leaf` → node tidak dibuat (`None`).
2. Node jadi daun bila salah satu benar:
   `len(node) < min_samples_split`, atau (klasifikasi saja)
   `max(distr) >= sum(distr) * sufficient_majority`, atau
   `max_depth is not None and level > max_depth`.
3. Selain itu cari split (Klaim 2); bila semua atribut ditolak, node tetap daun.

Bukti dengan setting default widget:

| Data | Node | Depth | min \|node\| | min \|internal\| | Mayoritas maks di internal |
|---|---|---|---|---|---|
| iris | 9 | 4 | 3 | 6 | 0.9074 |
| heart_disease | 77 | 11 | 2 | 5 | 0.9231 |
| heart_disease reg | 211 | 15 | 2 | 5 | (tidak berlaku) |

- `max_depth` = jumlah level split: `max_depth` 1, 2, 3 → `depth()` 1, 2, 3
  (heart_disease).
- `sufficient_majority` tidak ada di regresi.
- Semua checkbox mati → `max_depth=None, min_samples_split=2, min_samples_leaf=1,
  sufficient_majority=1`; dengan `sufficient_majority=1` hanya node murni yang berhenti
  karena mayoritas. Iris: semua daun murni, depth 5.

### Klaim 7. Bentuk model dan output

- `TreeModel` (`tree.py:136`) menyimpan `root` dan `instances`. Tipe node
  (`tree.py:15-133`): `Node` (daun), `DiscreteNode`, `MappedDiscreteNode`,
  `NumericNode` (punya `threshold`). Tiap node punya `attr`, `attr_idx`, `children`
  (boleh berisi `None`), `subset` (indeks baris latih), `value`, dan setelah kompilasi
  `parent`, `condition`, `description`.
- `value` klasifikasi = **jumlah per kelas** (bukan proporsi), juga di node internal.
  Iris akar `[50, 50, 50]`.
- `value` regresi = `[mean, var]` dengan var ddof=0 (`regression/tree.py:155-156`).
  heart_disease, akar target `max HR`: `[149.6073, 521.5388]`.
- Prediksi klasifikasi: probabilitas = `value` daun / jumlahnya, **tanpa smoothing**
  (`tree.py:208-218`); kelas = argmax. Prediksi regresi = mean daun. Prediksi memakai
  kode terkompilasi (`_compile`, `tree.py:291-361`; `compute_predictions` di Cython),
  hasilnya sama persis dengan penelusuran Python `get_values_by_nodes`: selisih maks
  0.0 di iris, heart_disease (dengan NaN), dan regresi.
- Iris dengan setting default: 9 node, 5 daun, depth 4, tipe node `Node` dan
  `NumericNode`.

## Tabel klaim

| No | Klaim singkat | Bukti utama |
|---|---|---|
| 1 | Widget → `Fitter` → `classification`/`regression.tree.TreeLearner`; bukan sklearn | `check_klaim1` |
| 2 | Info gain (bit) / penurunan MSE; seri → indeks atribut terkecil | `check_klaim2`, brute-force |
| 3 | Diskret: multiway atau 2-grup exhaustive (≤ 16 nilai) | `check_klaim3` |
| 4 | Threshold = maks sisi kiri (nilai data), x ≤ t kiri | `check_klaim4`, 106 node |
| 5 | Target NaN dibuang; atribut NaN tidak turun ke anak, skor × fraksi diketahui, prediksi berhenti | `check_klaim5` |
| 6 | Urutan stop: leaf → split → majority → depth | `check_klaim6` |
| 7 | `TreeModel`, value = jumlah kelas / [mean, var], prob tanpa smoothing | `check_klaim7` |

## Status

### Handoff 26 Nov 2026 (T2b-1)

**Selesai**

- Membaca `modelling/tree.py`, `classification/tree.py`, `regression/tree.py`,
  `tree.py`, `widgets/model/owtree.py`, dan `Fitter` di `modelling/base.py`.
- Dokumen ini plus `docs/proofs/tree_internals_check.py`: 7 klaim, semuanya `assert`,
  dijalankan dengan `D:\Orange\python.exe` → exit 0, baris terakhir
  `SEMUA KLAIM TERBUKTI`.
- Belum ada kode zestcore (sesuai brief).

**Temuan yang perlu diketahui sebelum desain ExtDT**

1. Threshold Orange adalah nilai data (maks sisi kiri), bukan titik tengah seperti
   sklearn (Klaim 4). Ini konvensi yang harus diputuskan (ADR) bila ExtDT ingin
   identik dengan Orange.
2. Seri skor diputus oleh urutan atribut (Klaim 2). Seri di dalam satu atribut
   (dua threshold dengan skor sama) belum dibuktikan arah pemutusnya, karena ada di
   Cython.
3. Default `binarize` learner (`False`) berbeda dengan default widget (`True`).
4. Preprocessor dari sinyal widget dibuang di jalur regresi (Klaim 1).

**Pertanyaan terbuka (untuk Raditya / Prof. Irfan)**

- ExtDT (extended decision tree) dari daftar rujukan: kriteria split-nya apa? Kalau
  bukan information gain, "identik dengan Orange" bukan target yang tepat, dan
  pembandingnya cukup akurasi dengan setting yang disamakan.
- Widget Tree di GUI dengan atribut > 16 nilai dan "Induce binary tree" ON: apakah GUI
  memberi peringatan/fallback, atau error? `owtree.py` 3.40.0 tidak punya kode
  fallback; belum dicek di GUI.
- Pemutus seri di dalam satu atribut (lihat temuan 2): perlu bila ExtDT harus identik
  1:1 pada data dengan banyak nilai kembar.

**Task berikutnya (usulan, jadwal diatur PM)**

- Putuskan target ExtDT: identik dengan Orange Tree, atau hanya sebanding.
- Bila identik: ADR konvensi threshold, pemutus seri, dan penanganan NaN, lalu
  `zestcore` tree mulai dari kasus iris (numerik murni, tanpa NaN).
