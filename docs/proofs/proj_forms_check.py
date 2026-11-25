"""Bukti pilihan bentuk objek proj untuk ZestPCAModel (ADR 0003, Catatan implementasi).

Jalankan dengan Python bawaan Orange:
    D:\\Orange\\python.exe docs\\proofs\\proj_forms_check.py

A: ZestPCA sendiri jadi proj (pola LinearProjector, Orange/projection/base.py:251-259).
B: objek kecil terpisah yang dibuat baru di setiap fit (pola PCA → objek sklearn,
   Orange/projection/pca.py:51-53). Bentuk B yang dipakai di orangecontrib/zest/pca.py.

Dua hal dicek pada iris (Continuize → SklImpute):
1. Projector yang sama di-fit ulang pada iris[:50]: apakah model dari fit pertama ikut
   berubah?
2. Pickle domain hasil (yang disimpan Save Data .pkl) dan pickle model.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from Orange.data import Table
from Orange.preprocess import Continuize, SklImpute
from Orange.projection import Projector, DomainProjection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import zestcore.pca as zp


class ZestPCAModel(DomainProjection):
    var_prefix = "PC"


# ---------- Bentuk A: projector = proj ----------
class ZestPCA_A(Projector):
    name = "Zest PCA"
    preprocessors = [Continuize(), SklImpute()]

    def fit(self, X, Y=None):
        r = zp.pca(X, n_components=100)
        self.mean_, self.components_ = r.mean, r.components
        self.explained_variance_ = r.explained_variance
        self.explained_variance_ratio_ = r.explained_variance_ratio
        return ZestPCAModel(self, self.domain, len(r.components))

    def transform(self, X):
        return zp.transform(X, self.mean_, self.components_)


# ---------- Bentuk B: objek kecil terpisah ----------
class ZestPCAFit:
    def __init__(self, r):
        self.mean_, self.components_ = r.mean, r.components
        self.explained_variance_ = r.explained_variance
        self.explained_variance_ratio_ = r.explained_variance_ratio

    def transform(self, X):
        return zp.transform(X, self.mean_, self.components_)


class ZestPCA_B(Projector):
    name = "Zest PCA"
    preprocessors = [Continuize(), SklImpute()]

    def fit(self, X, Y=None):
        r = zp.pca(X, n_components=100)
        return ZestPCAModel(ZestPCAFit(r), self.domain, len(r.components))


def main():
    iris = Table("iris")
    for label, cls in [("A", ZestPCA_A), ("B", ZestPCA_B)]:
        print(f"===== Bentuk {label} =====")
        projector = cls()
        model = projector(iris)
        print("type(model.proj)          :", type(model.proj).__name__)
        print("model.proj is projector   :", model.proj is projector)
        print("atribut model dari proj   :",
              sorted(k for k in vars(model) if k.endswith("_") or "tls" in k))
        before = model(iris).X.copy()

        # 1) Fit ulang projector yang SAMA pada data lain (setosa saja, 50 baris)
        projector(iris[:50])
        after = model(iris).X  # model PERTAMA dihitung ulang
        print("model 1 berubah setelah refit? maks |selisih| =",
              np.max(np.abs(after - before)))
        print("  model.components_ (salinan saat init) == model.proj.components_ ?",
              np.array_equal(model.components_, model.proj.components_))

        # 2) Pickle domain hasil (yang disimpan Save Data .pkl) dan model
        for what, obj in [("domain hasil", model(iris).domain), ("model", model)]:
            try:
                blob = pickle.dumps(obj)
                pickle.loads(blob)
                print(f"pickle {what:12s}: OK, {len(blob):6d} byte")
            except Exception as e:
                print(f"pickle {what:12s}: GAGAL -> {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
