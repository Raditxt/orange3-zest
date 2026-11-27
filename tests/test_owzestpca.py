"""Widget OWZestPCA vs OWPCA (T2a langkah 4b, rancangan di docs/orange-pca-internals.md).

Pembanding: widget PCA Orange 3.40.0 dengan setting yang sama. Komponen dan skor
dibandingkan dengan compare() (aturan seri/tanda dan ruang nol, toleransi spec
rtol=1e-5 atol=1e-8); rasio varians dengan compare_variance().
"""
from Orange.data import Table
from Orange.widgets.unsupervised.owpca import OWPCA
from orangewidget.tests.base import WidgetTest

from pca_compare import compare, compare_variance, zero_variance_count
from orangecontrib.zest.widgets.owzestpca import OWZestPCA


def names(domain):
    """(atribut, kelas, meta) sebagai nama."""
    return tuple([v.name for v in part]
                 for part in (domain.attributes, domain.class_vars, domain.metas))


class TestOWZestPCA(WidgetTest):
    def setUp(self):
        # self.widget mengaktifkan cek bawaan WidgetTest (ukuran minimum, ekspor
        # gambar, kelas dasar Msg)
        self.widget = self.create_widget(OWZestPCA)

    def run_owpca(self, data, settings):
        ref = self.create_widget(OWPCA, stored_settings=settings)
        self.send_signal(ref.Inputs.data, data, widget=ref)
        self.wait_until_finished(widget=ref)
        return (self.get_output(ref.Outputs.transformed_data, widget=ref),
                self.get_output(ref.Outputs.components, widget=ref))

    def run_zest(self, data, settings):
        got = self.create_widget(OWZestPCA, stored_settings=settings)
        self.send_signal(got.Inputs.data, data, widget=got)
        return (self.get_output(got.Outputs.transformed_data, widget=got),
                self.get_output(got.Outputs.components, widget=got))

    def test_output_iris_sama_dengan_owpca(self):
        data = Table("iris")
        n_null = zero_variance_count(data)
        for settings in ({"ncomponents": 2, "normalize": True},
                         {"ncomponents": 4, "normalize": False}):
            with self.subTest(**settings):
                ref_t, ref_c = self.run_owpca(data, settings)
                got_t, got_c = self.run_zest(data, settings)

                # Struktur: nama PC, kelas/meta asli, "variance" per PC, kolom Components
                # (nama saja: Variable.__eq__ ikut membandingkan compute_value)
                self.assertEqual(names(got_t.domain), names(ref_t.domain))
                self.assertEqual(
                    [a.attributes["variance"] for a in got_t.domain.attributes],
                    [a.attributes["variance"] for a in ref_t.domain.attributes])
                self.assertEqual(names(got_c.domain), names(ref_c.domain))
                self.assertEqual(list(got_c.metas[:, 0]), list(ref_c.metas[:, 0]))

                rows = compare(ref_c.X, ref_t.X, got_c.X, got_t.X, n_null)
                rows += compare_variance(ref_c.metas[:, 1].astype(float),
                                         got_c.metas[:, 1].astype(float), n_null)
                self.assertEqual([r for r in rows if not r["allclose"]], [])

    def test_data_none_mengosongkan_output(self):
        w = self.widget
        self.send_signal(w.Inputs.data, Table("iris"))
        self.assertIsNotNone(self.get_output(w.Outputs.transformed_data))

        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.transformed_data))
        self.assertIsNone(self.get_output(w.Outputs.components))
