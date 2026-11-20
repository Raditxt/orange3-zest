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


if __name__ == "__main__":
    check_klaim1()
    print()
    check_klaim2()
    print()
    check_klaim3()
    print()
    check_klaim4()
