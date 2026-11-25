"""Adaptor Orange untuk zestcore.pca: pasangan Projector/DomainProjection seperti
PCA/PCAModel Orange (Orange/projection/pca.py). Desain: docs/adr/0003 (opsi 1c,
Catatan implementasi). Semua kelas di level modul supaya domain hasil bisa di-pickle."""
from Orange.data.util import get_unique_names
from Orange.preprocess import Continuize, SklImpute
from Orange.projection import DomainProjection, Projector

from zestcore.pca import pca, transform


class ZestPCAFit:
    """Hasil satu fit, dibuat baru di setiap ZestPCA.fit() dan tidak diubah sesudahnya.
    Berperan seperti objek sklearn PCA di dalam PCAModel (model.proj)."""

    def __init__(self, result):
        self.mean_ = result.mean
        self.components_ = result.components
        self.explained_variance_ = result.explained_variance
        self.explained_variance_ratio_ = result.explained_variance_ratio

    def transform(self, X):
        return transform(X, self.mean_, self.components_)


class ZestPCAModel(DomainProjection):
    var_prefix = "PC"

    def _get_var_names(self, n):
        # PC1, PC2, ... seperti PCAModel (Orange/projection/pca.py:76-78)
        names = [f"{self.var_prefix}{i}" for i in range(1, n + 1)]
        return get_unique_names(self.orig_domain, names)


class ZestPCA(Projector):
    """PCA zestcore lewat preprocessor Orange (ADR 0002). Preprocessor default sama
    dengan PCA Orange; widget menambahkan Normalize(center=False) bila dicentang
    (owpca.py:281-282). Widget mengirim n_components=100 (owpca.py:27)."""
    name = "Zest PCA"
    preprocessors = [Continuize(), SklImpute()]

    def __init__(self, n_components=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.n_components = n_components

    def fit(self, X, Y=None):
        fitted = ZestPCAFit(pca(X, n_components=self.n_components))
        return ZestPCAModel(fitted, self.domain, len(fitted.components_))
