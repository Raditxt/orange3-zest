"""Widget Zest PCA: membungkus ZestPCA (orangecontrib/zest/pca.py). Output disusun
seperti OWPCA (Orange/widgets/unsupervised/owpca.py:316-355). Rancangan:
docs/orange-pca-internals.md, "Rancangan OWZestPCA"."""
import warnings

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.data.util import get_unique_names
from Orange.preprocess import Normalize
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.zest.pca import ZestPCA
from zestcore.pca import ShapeWarning

MAX_COMPONENTS = 100  # sama dengan OWPCA (owpca.py:27)


class OWZestPCA(OWWidget):
    name = "Zest PCA"
    description = "PCA NumPy murni (zestcore), hasilnya identik dengan widget PCA Orange."
    keywords = "pca, principal component analysis, zest"
    icon = "icons/ZestPCA.svg"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        transformed_data = Output("Transformed Data", Table)
        components = Output("Components", Table, dynamic=False)

    ncomponents = Setting(2)
    normalize = Setting(True)

    want_main_area = False

    class Warning(OWWidget.Warning):
        shape_outside_spec = Msg("{}")  # teks ShapeWarning dari zestcore

    class Error(OWWidget.Error):
        fit_failed = Msg("{}")

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
        self._model = self._transformed = None
        self.Warning.shape_outside_spec.clear()
        self.Error.fit_failed.clear()
        if self.data is None:
            self.clear_outputs()
            return
        preprocessors = list(ZestPCA.preprocessors)
        if self.normalize:
            preprocessors.append(Normalize(center=False))  # owpca.py:281-282
        # catch_warnings mengubah state global: alasan fit dibuat sinkron
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ShapeWarning)
            try:
                model = ZestPCA(n_components=MAX_COMPONENTS,
                                preprocessors=preprocessors)(self.data)
                # Skor lewat compute_value, jalur yang sama dengan OWPCA (owpca.py:321)
                transformed = model(self.data)
            except ValueError as err:  # misalnya zestcore menolak n < 2
                self.Error.fit_failed(str(err))
                model = transformed = None
        self._show_warnings(caught)
        self._model, self._transformed = model, transformed
        self.commit()

    def _show_warnings(self, caught):
        """ShapeWarning jadi Msg widget; warning lain dikeluarkan lagi apa adanya."""
        for w in caught:
            if issubclass(w.category, ShapeWarning):
                self.Warning.shape_outside_spec(str(w.message))
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    def clear_outputs(self):
        self.Outputs.transformed_data.send(None)
        self.Outputs.components.send(None)

    def commit(self):
        """Iris hasil fit ke ncomponents, tanpa fit ulang (owpca.py:316-355)."""
        if self._model is None:  # spin diubah saat belum ada hasil fit
            self.clear_outputs()
            return
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
