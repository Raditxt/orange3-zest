"""Widget Zest PCA: membungkus ZestPCA (orangecontrib/zest/pca.py). Output disusun
seperti OWPCA (Orange/widgets/unsupervised/owpca.py:316-355). Rancangan:
docs/orange-pca-internals.md, "Rancangan OWZestPCA"."""
import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.data.util import get_unique_names
from Orange.preprocess import Normalize
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from orangecontrib.zest.pca import ZestPCA

MAX_COMPONENTS = 100  # sama dengan OWPCA (owpca.py:27)


class OWZestPCA(OWWidget):
    name = "Zest PCA"
    description = "PCA NumPy murni (zestcore), hasilnya identik dengan widget PCA Orange."
    keywords = "pca, principal component analysis, zest"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        transformed_data = Output("Transformed Data", Table)
        components = Output("Components", Table, dynamic=False)

    ncomponents = Setting(2)
    normalize = Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None
        self._model = None        # ZestPCAModel hasil fit terakhir
        self._transformed = None  # model(data), semua komponen

        box = gui.widgetBox(self.controlArea, "Components Selection")
        gui.spin(box, self, "ncomponents", 1, MAX_COMPONENTS,
                 label="Components:", callback=self.commit)
        gui.checkBox(self.controlArea, self, "normalize", "Normalize variables",
                     callback=self.fit)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.fit()

    def fit(self):
        """Fit ulang (data atau normalize berubah), lalu kirim output."""
        preprocessors = list(ZestPCA.preprocessors)
        if self.normalize:
            preprocessors.append(Normalize(center=False))  # owpca.py:281-282
        self._model = ZestPCA(n_components=MAX_COMPONENTS,
                              preprocessors=preprocessors)(self.data)
        # Skor lewat compute_value, jalur yang sama dengan OWPCA (owpca.py:321)
        self._transformed = self._model(self.data)
        self.commit()

    def commit(self):
        """Iris hasil fit ke ncomponents, tanpa fit ulang (owpca.py:316-355)."""
        k = self.ncomponents
        ratio = self._model.explained_variance_ratio_[:k]

        attrs = self._transformed.domain.attributes[:k]
        for var, r in zip(attrs, ratio):
            var.attributes["variance"] = round(r, 6)
        domain = Domain(attrs, self.data.domain.class_vars, self.data.domain.metas)
        transformed = self._transformed.from_table(domain, self._transformed)

        # compute_value=lambda: Components tidak dihitung ulang dari data lain
        proposed = [a.name for a in self._model.orig_domain.attributes]
        meta_vars = [StringVariable(get_unique_names(proposed, "components")),
                     ContinuousVariable(get_unique_names(proposed, "variance"))]
        metas = np.column_stack(
            [np.array([f"PC{i + 1}" for i in range(len(ratio))], dtype=object),
             ratio.astype(object)])
        dom = Domain([ContinuousVariable(name, compute_value=lambda _: None)
                      for name in proposed], metas=meta_vars)
        components = Table(dom, self._model.components_[:k], metas=metas)
        components.name = "components"

        self.Outputs.transformed_data.send(transformed)
        self.Outputs.components.send(components)
