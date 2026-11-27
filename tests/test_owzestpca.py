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

    def test_bentuk_di_luar_spec_tampil_sebagai_warning(self):
        """heart_disease[:100]: 100x25 setelah Continuize, n < 10p (ADR 0001)."""
        w = self.widget
        self.send_signal(w.Inputs.data, Table("heart_disease")[:100])
        self.assertTrue(w.Warning.shape_outside_spec.is_shown())
        self.assertIn("100x25", str(w.Warning.shape_outside_spec))
        self.assertIsNotNone(self.get_output(w.Outputs.transformed_data))

        self.send_signal(w.Inputs.data, Table("iris"))
        self.assertFalse(w.Warning.shape_outside_spec.is_shown())

    def test_fit_gagal_tampil_sebagai_error(self):
        """iris[:1]: zestcore.pca menolak n < 2 dengan ValueError."""
        w = self.widget
        iris = Table("iris")
        self.send_signal(w.Inputs.data, iris)
        self.send_signal(w.Inputs.data, iris[:1])
        self.assertTrue(w.Error.fit_failed.is_shown())
        self.assertIn("minimal 2 baris", str(w.Error.fit_failed))
        self.assertIsNone(self.get_output(w.Outputs.transformed_data))
        self.assertIsNone(self.get_output(w.Outputs.components))

        self.send_signal(w.Inputs.data, iris)
        self.assertFalse(w.Error.fit_failed.is_shown())
        self.assertIsNotNone(self.get_output(w.Outputs.transformed_data))
