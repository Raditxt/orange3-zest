# Orange3-Zest

Add-on Orange Data Mining supaya model-model dari penelitian (terutama yang belum ada di Orange, misalnya metaheuristik) bisa dipakai pengguna awam lewat widget. Proyek magang Raditya dengan pembimbing Prof. Irfan. Repo publik, lisensi GPL-3.0.

## Tahap sekarang

Tahap 1 adalah proof of concept: PCA NumPy murni yang hasilnya identik 1:1 dengan PCA bawaan Orange, dibuktikan di notebook. Widget PCA TIDAK dibuat. Tahap 2 adalah model pertama dari daftar rujukan Prof. Irfan (Firefly, Komodo Mlipir, Al-Biruni Earth Radius, PSO, extended decision tree), dalam bentuk widget.

## Struktur dua lapis (aturan keras)

- `zestcore/`: inti algoritma, NumPy murni. DILARANG `import Orange` atau `import sklearn` di sini.
- `orangecontrib/zest/`: lapis widget Orange, hanya membungkus `zestcore`. `orangecontrib/` adalah namespace package, jangan tambahkan `__init__.py` langsung di dalamnya.
- `tests/`: pytest. scikit-learn dan Orange boleh dipakai di sini sebagai pembanding.
- `notebooks/`, `docs/`, `evidence/` (CSV + .ows dari GUI Orange).
- Spesifikasi PCA: docs/orange-pca-internals.md; keputusan desain: docs/adr/.

## Lingkungan

Semua pekerjaan dijalankan dengan Python bawaan Orange supaya angkanya identik dengan GUI:

- Interpreter: `D:\Orange\python.exe`
- Python 3.12.12, Orange 3.40.0, scikit-learn 1.5.2, NumPy 2.1.3
- Kode sumber Orange yang terpasang: `D:\Orange\Lib\site-packages\Orange\`
- Shell: PowerShell (Windows)

Rujuk kode sumber dari versi yang terpasang di atas, bukan dari ingatan atau branch master GitHub.

## Prinsip teknis

- "Identik" harus dibuktikan dengan angka: `np.allclose` plus selisih maksimum yang dilaporkan. Selalu sebutkan dibanding apa, toleransinya berapa, dan setting-nya apa.
- Tanda komponen PCA bisa terbalik antar implementasi. Pakai konvensi tanda yang deterministik dan tulis di dokumentasi.
- Kalau ternyata tidak bisa identik, itu temuan yang sah. Jelaskan penyebabnya, jangan disembunyikan atau ditutupi dengan toleransi longgar.
- Sederhana dulu: perubahan kecil, setiap langkah bisa dicek, tanpa abstraksi yang belum dibutuhkan.

## Cara bekerja dengan Raditya

- Raditya menargetkan pembagian kerja 50:50 dengan AI, dan harus paham serta bisa mempertanggungjawabkan setiap baris kode di depan pembimbing.
- Sebelum menulis kode, jelaskan dulu pendekatannya secara singkat. Untuk materi baru (SVD, PCA, arsitektur widget Orange), beri contoh kecil yang bisa langsung dijalankan.
- Kerjakan per langkah kecil. Setelah setiap langkah, jelaskan apa yang berubah dan kenapa, lalu tunggu Raditya sebelum lanjut.
- Laporan magang (BAB I–V) ditulis Raditya sendiri. Boleh membantu menyusun kerangka dan mengecek akurasi teknis, tapi jangan menulis laporannya secara utuh.
- Bahasa: Indonesia, lugas.
