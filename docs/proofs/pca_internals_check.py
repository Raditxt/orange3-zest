"""Bukti 7 klaim tentang cara Orange 3.40.0 menjalankan PCA.

Jalankan dengan Python bawaan Orange:
    D:\\Orange\\python.exe docs\\proofs\\pca_internals_check.py

Setiap klaim punya satu fungsi check_klaimN(). Spesifikasi hasilnya ada di
docs/orange-pca-internals.md.
"""
import Orange
from Orange.data import Table
from Orange.preprocess import Continuize, SklImpute, Normalize
from Orange.projection import PCA
from Orange.widgets.unsupervised.owpca import OWPCA


def check_klaim1():
    print("=== Klaim 1: preprocessor default PCA dan tambahan dari widget ===")
    print("Orange versi:", Orange.version.version)

    # (a) Preprocessor default di level kelas PCA (diwarisi dari SklProjector)
    print("PCA.preprocessors      :", [type(p).__name__ for p in PCA.preprocessors])

    # (b) Projector yang dibuat widget, tanpa membuka GUI.
    #     _create_projector() hanya membaca self.normalize dan self.ncomponents,
    #     jadi cukup objek tiruan yang punya dua atribut itu.
    class Palsu:
        normalize = True     # default widget: Setting(True)
        ncomponents = 2

    proj = OWPCA._create_projector(Palsu())
    print("widget, normalize=True :", [type(p).__name__ for p in proj.preprocessors])
    norm = proj.preprocessors[-1]
    print("  Normalize terakhir   : center =", norm.center,
          "| norm_type =", norm.norm_type)

    Palsu.normalize = False
    proj = OWPCA._create_projector(Palsu())
    print("widget, normalize=False:", [type(p).__name__ for p in proj.preprocessors])


def check_klaim2():
    import numpy as np
    from Orange.data import Domain, ContinuousVariable, DiscreteVariable

    print("=== Klaim 2: Continuize default = Indicators, zero_based ===")
    c = Continuize()
    print("default: multinomial_treatment =", c.multinomial_treatment,
          "| zero_based =", c.zero_based)

    # Tabel mini: 1 kontinu, 1 biner, 1 diskrit 3 nilai, 1 diskrit 1 nilai.
    # Nilai diskrit disimpan sebagai indeks (0, 1, 2); NaN = kosong.
    domain = Domain([
        ContinuousVariable("umur"),
        DiscreteVariable("jk", values=("L", "P")),               # biner
        DiscreteVariable("warna", values=("merah", "hijau", "biru")),
        DiscreteVariable("konstan", values=("x",)),              # 1 nilai
    ])
    X = np.array([
        [20, 0, 0,      0],
        [30, 1, 1,      0],
        [40, 1, 2,      0],
        [50, 0, np.nan, 0],   # warna kosong
    ], dtype=float)
    data = Table.from_numpy(domain, X)

    out = c(data)
    print("kolom sebelum:", [v.name for v in data.domain.attributes])
    print("kolom sesudah:", [v.name for v in out.domain.attributes])
    print(out.X)


def check_klaim3():
    import numpy as np

    print("=== Klaim 3: Normalize pakai ddof=0, kovarians PCA pakai ddof=1 ===")
    iris = Table("iris")
    X = iris.X                                    # 150 x 4, tanpa NaN

    # (a) Normalize(center=False): X dibagi SD saja
    Xn = Normalize(center=False)(iris).X
    ref0 = X / X.std(axis=0, ddof=0)              # SD populasi (bagi n)
    ref1 = X / X.std(axis=0, ddof=1)              # SD sampel (bagi n-1)
    print("Normalize vs X/std(ddof=0): allclose =", np.allclose(Xn, ref0),
          "| max|selisih| =", np.abs(Xn - ref0).max())
    print("Normalize vs X/std(ddof=1): allclose =", np.allclose(Xn, ref1),
          "| max|selisih| =", np.abs(Xn - ref1).max())

    # (b) PCA Orange (tanpa Normalize) pada iris: explained_variance_
    #     dibanding eigenvalue kovarians ddof=1 dan ddof=0
    model = PCA(n_components=4, random_state=0)(iris)
    ev = model.proj.explained_variance_
    eig1 = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=1)))[::-1]
    eig0 = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=0)))[::-1]
    print("explained_variance_      :", ev)
    print("vs eig cov ddof=1: allclose =", np.allclose(ev, eig1),
          "| max|selisih| =", np.abs(ev - eig1).max())
    print("vs eig cov ddof=0: allclose =", np.allclose(ev, eig0),
          "| max|selisih| =", np.abs(ev - eig0).max())

    # (c) Dampak salah ddof di Normalize: skala seragam sqrt((n-1)/n)
    #     -> components_ dan ratio sama, explained_variance_ dan skor beda.
    from sklearn.decomposition import PCA as SkPCA
    a = SkPCA().fit(ref0)
    b = SkPCA().fit(ref1)
    print("[salah ddof] components_ sama   :", np.allclose(a.components_, b.components_))
    print("[salah ddof] ratio sama         :", np.allclose(a.explained_variance_ratio_,
                                                          b.explained_variance_ratio_))
    print("[salah ddof] explained_var sama :", np.allclose(a.explained_variance_,
                                                          b.explained_variance_),
          "| rasio b/a =", (b.explained_variance_ / a.explained_variance_)[0],
          "| (n-1)/n =", (len(X) - 1) / len(X))
    print("[salah ddof] skor PC sama       :", np.allclose(a.transform(ref0),
                                                          b.transform(ref1)),
          "| max|selisih| =", np.abs(a.transform(ref0) - b.transform(ref1)).max())


def _widget_projector():
    """Projector persis seperti widget PCA dengan setting default."""
    class Palsu:
        normalize = True
        ncomponents = 2
    return OWPCA._create_projector(Palsu())


def check_klaim4():
    import numpy as np

    print("=== Klaim 4: solver yang dipilih svd_solver='auto' ===")
    import sklearn
    print("scikit-learn versi:", sklearn.__version__)
    for nama in ["iris", "heart_disease"]:
        data = Table(nama)
        model = _widget_projector()(data)       # jalur widget, Normalize ON
        skl = model.proj                        # objek sklearn PCA yang sudah fit
        n, p = skl.n_samples_, skl.n_features_in_
        print(f"{nama:14s} X setelah preprocessing = {n} x {p} | "
              f"n >= 10p? {n >= 10 * p} (10p = {10 * p}) | "
              f"n_components = {skl.n_components_} | "
              f"_fit_svd_solver = {skl._fit_svd_solver}")
    print("bank marketing: belum ada di repo, dicek di task bank marketing")

    # Catatan dari Klaim 2: tiap variabel diskrit menyumbang 1 arah varians ~0
    model = _widget_projector()(Table("heart_disease"))
    ev = model.proj.explained_variance_
    np.set_printoptions(precision=4, linewidth=100)
    print("heart_disease explained_variance_:")
    print(ev)
    print("jumlah komponen < 1e-10:", int((ev < 1e-10).sum()),
          "| nilai terkecil =", ev.min())


def check_klaim5():
    import numpy as np

    print("=== Klaim 5: tanda komponen lewat svd_flip(u_based_decision=False) ===")
    np.set_printoptions(precision=4, linewidth=100)

    # (a) Tiap baris components_: elemen dengan |nilai| terbesar harus positif.
    #     Selisih |terbesar| - |kedua terbesar| dicetak untuk melihat apakah ada
    #     baris yang nyaris seri (argmax memilih indeks pertama kalau seri).
    for nama in ["iris", "heart_disease"]:
        C = _widget_projector()(Table(nama)).proj.components_
        idx = np.abs(C).argmax(axis=1)
        tanda = C[np.arange(len(C)), idx]
        urut = np.sort(np.abs(C), axis=1)
        jarak = urut[:, -1] - urut[:, -2]
        print(f"{nama:14s} baris = {len(C)} | baris dengan elemen |terbesar| > 0: "
              f"{int((tanda > 0).sum())} | jarak |terbesar|-|kedua| minimum = "
              f"{jarak.min():.3e} (baris {jarak.argmin() + 1})")

    # (b) heart_disease: selisih eigenvalue berurutan. Eigenvalue kembar
    #     berarti vektor eigennya tidak unik (bebas diputar di ruang itu).
    ev = _widget_projector()(Table("heart_disease")).proj.explained_variance_
    gap = ev[:-1] - ev[1:]
    print("heart_disease selisih eigenvalue k - (k+1), k = 1..18:")
    print(gap[:18])
    print("selisih terkecil di k = 1..18 =", f"{gap[:18].min():.3e}",
          "| eigenvalue ke-18 =", f"{ev[17]:.3e}", "| ke-19 =", f"{ev[18]:.3e}")

    # (c) Bandingkan covariance_eigh (pilihan 'auto') dengan 'full' (SVD LAPACK).
    #     Preprocessing dan setting lain sama persis dengan widget.
    data = Table("heart_disease")
    pre = PCA.preprocessors + [Normalize(center=False)]
    eigh = PCA(n_components=100, random_state=0, preprocessors=pre)(data).proj
    full = PCA(n_components=100, random_state=0, svd_solver="full",
               preprocessors=pre)(data).proj
    print("solver:", eigh._fit_svd_solver, "vs", full._fit_svd_solver)
    a, b = eigh.components_, full.components_
    print("components_ 1-18 : allclose =", np.allclose(a[:18], b[:18]),
          "| max|selisih| =", f"{np.abs(a[:18] - b[:18]).max():.3e}")
    print("components_ 19-25: allclose =", np.allclose(a[18:], b[18:]),
          "| max|selisih| =", f"{np.abs(a[18:] - b[18:]).max():.3e}")
    print("explained_variance_ 1-18: allclose =",
          np.allclose(eigh.explained_variance_[:18], full.explained_variance_[:18]),
          "| max|selisih| =",
          f"{np.abs(eigh.explained_variance_[:18] - full.explained_variance_[:18]).max():.3e}")

    # (d) Baris yang nyaris seri: dua elemen |terbesar| sama. Nama kolom diambil
    #     dari data setelah preprocessing widget.
    proj = _widget_projector()
    pre = data
    for pp in proj.preprocessors:
        pre = pp(pre)
    nama_kolom = [v.name for v in pre.domain.attributes]
    C0 = proj(data).proj.components_
    urut = np.sort(np.abs(C0), axis=1)
    jarak = urut[:, -1] - urut[:, -2]
    for k in np.where(jarak < 1e-10)[0]:
        i, j = np.argsort(-np.abs(C0[k]))[:2]
        print(f"seri baris {k + 1:2d} (jarak {jarak[k]:.1e}): "
              f"{nama_kolom[i]} = {C0[k, i]:+.15f}, {nama_kolom[j]} = {C0[k, j]:+.15f}")

    # (e) Data sama, urutan baris diacak (20 kali): komponen 1-18 mana yang
    #     tandanya terbalik dibanding urutan asli?
    rng = np.random.default_rng(0)
    balik = np.zeros(18, dtype=int)
    for _ in range(20):
        C = _widget_projector()(data[rng.permutation(len(data))]).proj.components_
        balik += np.array([C[k] @ C0[k] < 0 for k in range(18)])
    print("acak baris 20x, komponen 1-18 yang tandanya pernah terbalik:",
          {k + 1: int(balik[k]) for k in range(18) if balik[k] > 0})


def check_klaim6():
    import numpy as np
    from sklearn.decomposition import PCA as SkPCA

    print("=== Klaim 6: n_components=100, random_state=0, dipotong ke min(X.shape) ===")

    # (a) Parameter projector yang dibuat widget
    proj = _widget_projector()
    print("widget: n_components =", proj.params["n_components"],
          "| random_state =", proj.params["random_state"],
          "| svd_solver =", proj.params["svd_solver"],
          "| projector.component =", proj.component)

    # (b) Pemotongan min(min(X.shape), 100). X = data SETELAH preprocessing.
    iris = Table("iris")
    for nama, data in [("iris", iris), ("heart_disease", Table("heart_disease")),
                       ("iris 3 baris", iris[:3])]:
        skl = _widget_projector()(data).proj
        print(f"{nama:14s} X = {skl.n_samples_} x {skl.n_features_in_} | "
              f"n_components_ = {skl.n_components_} | "
              f"components_.shape = {skl.components_.shape} | "
              f"sum(ratio) = {skl.explained_variance_ratio_.sum():.15f}")

    # (c) Tanpa pemotongan Orange, sklearn menolak n_components=100
    try:
        SkPCA(n_components=100).fit(iris.X)
        print("sklearn n_components=100 pada iris: lolos")
    except ValueError as e:
        print("sklearn n_components=100 pada iris: ValueError:", str(e)[:70], "...")

    # (d) Pilihan jumlah komponen di widget (ncomponents) tidak mengubah fit
    class Lima:
        normalize = True
        ncomponents = 5
    a = _widget_projector()(Table("heart_disease")).proj
    b = OWPCA._create_projector(Lima())(Table("heart_disease")).proj
    print("ncomponents 2 vs 5: n_components_ =", a.n_components_, b.n_components_,
          "| components_ sama persis =", np.array_equal(a.components_, b.components_))

    # (e) random_state tidak dipakai jalur covariance_eigh / full
    for solver in ["auto", "full"]:
        pre = PCA.preprocessors + [Normalize(center=False)]
        r0 = PCA(n_components=100, random_state=0, svd_solver=solver,
                 preprocessors=pre)(Table("heart_disease")).proj
        r1 = PCA(n_components=100, random_state=12345, svd_solver=solver,
                 preprocessors=pre)(Table("heart_disease")).proj
        print(f"svd_solver={solver:4s} ({r0._fit_svd_solver}): random_state 0 vs 12345, "
              f"components_ sama persis = {np.array_equal(r0.components_, r1.components_)}")

    # (f) Batas 100 komponen: bila min(n, p) > 125 maka 100 < 0.8*min(n, p),
    #     dan 'auto' bisa memilih 'randomized' (_pca.py:539). Data sintetis normal.
    from Orange.data import Domain, ContinuousVariable
    rng = np.random.default_rng(0)
    for n, p in [(1000, 125), (1000, 126), (1000, 200), (600, 130)]:
        dom = Domain([ContinuousVariable(f"x{i}") for i in range(p)])
        skl = PCA(n_components=100, random_state=0)(
            Table.from_numpy(dom, rng.normal(size=(n, p)))).proj
        print(f"sintetis {n}x{p}: solver = {skl._fit_svd_solver} | "
              f"n_components_ = {skl.n_components_}")


def check_klaim7():
    import numpy as np

    print("=== Klaim 7: heart_disease, NaN, variabel diskrit, 7 arah varians ~0 ===")
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    data = Table("heart_disease")

    # (a) Data mentah: tipe tiap atribut dan jumlah NaN (target tidak ikut PCA)
    attrs = data.domain.attributes
    print("X mentah =", data.X.shape, "| target =", data.domain.class_var.name)
    for i, v in enumerate(attrs):
        jenis = f"diskrit, {len(v.values)} nilai" if v.is_discrete else "kontinu"
        print(f"  {v.name:26s} {jenis:18s} NaN = {int(np.isnan(data.X[:, i]).sum())}")
    diskrit = [v for v in attrs if v.is_discrete]
    print("total NaN =", int(np.isnan(data.X).sum()),
          "| baris dengan NaN =", int(np.isnan(data.X).any(axis=1).sum()),
          "| variabel diskrit =", len(diskrit),
          "| kontinu =", len(attrs) - len(diskrit))

    # (b) Kolom sebelum/sesudah Continuize: kontinu tetap 1 kolom,
    #     diskrit jadi 1 kolom per nilai
    con = Continuize()(data)
    harap = (len(attrs) - len(diskrit)) + sum(len(v.values) for v in diskrit)
    print("kolom: sebelum =", len(attrs), "| sesudah Continuize =",
          len(con.domain.attributes), "| harapan 6 + sum(jumlah nilai) =", harap)

    # (c) SklImpute: NaN diisi rata-rata kolom. Indikator thal di baris kosong
    #     jadi pecahan, dan jumlah indikator satu variabel tetap 1.
    imp = SklImpute()(con)
    nama = [v.name for v in imp.domain.attributes]
    print("kolom sesudah SklImpute =", len(nama), "| NaN tersisa =",
          int(np.isnan(imp.X).sum()))
    kosong = np.where(np.isnan(data.X[:, attrs.index(data.domain["thal"])]))[0]
    kol_thal = [j for j, s in enumerate(nama) if s.startswith("thal=")]
    print("baris thal kosong:", kosong + 1, "| kolom:", [nama[j] for j in kol_thal])
    print(imp.X[np.ix_(kosong, kol_thal)])
    print("rata-rata kolom thal (data lengkap):",
          np.nanmean(con.X[:, kol_thal], axis=0))
    # jumlah kolom indikator per variabel diskrit, semua baris
    grup = {v.name: [j for j, s in enumerate(nama) if s.startswith(v.name + "=")]
            for v in diskrit}
    jml = np.array([imp.X[:, g].sum(axis=1) for g in grup.values()])
    print("jumlah indikator per variabel = 1 di semua baris:",
          np.allclose(jml, 1), "| max|jumlah - 1| =", f"{np.abs(jml - 1).max():.1e}")

    # (d) Cocokkan dengan Klaim 4. Setelah Normalize, kolom z_j = x_j / sd_j.
    #     Karena sum_j x_j = 1 untuk tiap grup, vektor w (w_j = sd_j di kolom
    #     grup, 0 di luar) memberi z @ w = 1 konstan, jadi varians arah w = 0.
    model = _widget_projector()(data)
    ev, C = model.proj.explained_variance_, model.proj.components_
    nol = ev < 1e-10
    print("komponen dengan explained_variance_ < 1e-10:", int(nol.sum()),
          "| nomor:", np.where(nol)[0] + 1)
    sd = imp.X.std(axis=0, ddof=0)                  # SD yang dipakai Normalize
    W = np.zeros((len(grup), len(nama)))
    for r, g in enumerate(grup.values()):
        W[r, g] = sd[g]
    W /= np.linalg.norm(W, axis=1, keepdims=True)   # panjang 1
    # Proyeksi tiap w ke 18 komponen bervarians > 0 harus ~0,
    # dan sisa w setelah diproyeksikan ke 7 komponen ~0 harus ~0.
    ke_besar = np.abs(W @ C[~nol].T).max()
    Cn = C[nol]
    sisa = np.abs(W - (W @ Cn.T) @ Cn).max()
    print("max|w . komponen 1-18|      =", f"{ke_besar:.2e}")
    print("max|w - proyeksi ke 19-25|  =", f"{sisa:.2e}")
    print("rank W =", np.linalg.matrix_rank(W),
          "| varians data di arah w (ddof=1):",
          np.var(_preprocessed(data) @ W.T, axis=0, ddof=1))

    # (e) Rumus jumlah arah varians nol, diuji di data sintetis 50 baris.
    #     Continuize membuang variabel dari len(var.values) yang dideklarasikan
    #     (continuize.py:21), bukan dari nilai yang muncul. Normalize mengganti
    #     SD = 0 dengan 1 (normalize.py:53-54), jadi kolom konstan tetap konstan.
    from Orange.data import Domain, ContinuousVariable, DiscreteVariable
    rng = np.random.default_rng(0)
    n = 50
    kolom = {
        # nama: (variabel, isi kolom, prediksi rumus awal, prediksi koreksi)
        "k1": (ContinuousVariable("k1"), rng.normal(size=n), 0, 0),
        "k2": (ContinuousVariable("k2"), rng.normal(size=n), 0, 0),
        "konst": (ContinuousVariable("konst"), np.full(n, 7.0), 1, 1),
        # A: 3 nilai, semua muncul, 3 NaN
        "A": (DiscreteVariable("A", values=("a", "b", "c")),
              np.r_[rng.integers(0, 3, n - 3), [np.nan] * 3], 1, 1),
        # B: 3 nilai, hanya 2 muncul -> (3-2)+1 = 2
        "B": (DiscreteVariable("B", values=("a", "b", "c")),
              rng.integers(0, 2, n).astype(float), 1, 2),
        # C: 2 nilai, hanya 1 muncul -> (2-1)+1 = 2
        "C": (DiscreteVariable("C", values=("a", "b")), np.zeros(n), 1, 2),
        # D: 2 nilai, seluruhnya NaN -> dibuang SklImpute, 0
        "D": (DiscreteVariable("D", values=("a", "b")), np.full(n, np.nan), 0, 0),
        # E: 1 nilai -> dibuang Continuize, 0
        "E": (DiscreteVariable("E", values=("a",)), np.zeros(n), 0, 0),
    }
    sint = Table.from_numpy(Domain([v for v, *_ in kolom.values()]),
                            np.column_stack([x for _, x, *_ in kolom.values()]))
    ev = _widget_projector()(sint).proj.explained_variance_
    print(f"sintetis {n} baris: kolom setelah preprocessing =",
          _preprocessed(sint).shape[1], "| komponen =", len(ev),
          "| explained_variance_ < 1e-10 =", int((ev < 1e-10).sum()))
    print("  prediksi rumus awal (diskrit >=2 nilai, tidak seluruhnya NaN, + kontinu konstan) =",
          sum(a for *_, a, _ in kolom.values()))
    print("  prediksi koreksi (per diskrit: nilai deklarasi - nilai muncul + 1) =",
          sum(b for *_, b in kolom.values()))


def _preprocessed(data):
    """X setelah preprocessing widget (Continuize, SklImpute, Normalize)."""
    for pp in _widget_projector().preprocessors:
        data = pp(data)
    return data.X


if __name__ == "__main__":
    check_klaim1()
    print()
    check_klaim2()
    print()
    check_klaim3()
    print()
    check_klaim4()
    print()
    check_klaim5()
    print()
    check_klaim6()
    print()
    check_klaim7()
