"""Bandingkan dua hasil PCA menurut aturan uji docs/orange-pca-internals.md."""
import numpy as np
from Orange.widgets.unsupervised.owpca import OWPCA

RTOL, ATOL = 1e-5, 1e-8  # np.allclose default = toleransi spec
TIE_GAP = 1e-10          # aturan uji poin 3


def _widget_projector(normalize=True):
    """Projector widget PCA; default = setting default widget (Normalize ON).
    ncomponents widget hanya mengiris output (Klaim 6), jadi tidak berpengaruh di sini.
    OWPCA._create_projector adalah API privat Orange 3.40.0."""
    class Setting:
        pass
    Setting.normalize = normalize
    Setting.ncomponents = 2
    return OWPCA._create_projector(Setting())


def orange_widget_pca(table):
    """(proj, skor) lewat projector widget PCA.

    proj = objek sklearn PCA di dalam model Orange (components_, explained_variance_,
    explained_variance_ratio_); skor n x k. k = min(n, p, 100)."""
    model = _widget_projector()(table)
    return model.proj, model(table).X


def preprocessed_X(table):
    """X yang diterima sklearn di dalam Orange: preprocessor projector widget
    (Continuize -> SklImpute -> Normalize) dijalankan berurutan (ADR 0002)."""
    data = table
    for pp in _widget_projector().preprocessors:
        data = pp(data)
    return data.X


def tied_rows(components):
    """Indeks baris yang seri: |terbesar| - |kedua terbesar| < TIE_GAP."""
    top = -np.sort(-np.abs(components), axis=1)
    return [int(k) for k in np.flatnonzero(top[:, 0] - top[:, 1] < TIE_GAP)]


def zero_variance_count(table):
    """Batas bawah jumlah arah varians nol (aturan uji poin 2), dari data sebelum fit:
    sum(nilai deklarasi - nilai yang muncul + 1) per variabel diskrit yang tidak
    dibuang, + jumlah kolom kontinu konstan."""
    count = 0
    for var in table.domain.attributes:
        col = table.get_column(var)
        present = col[~np.isnan(col)]
        if present.size == 0:          # seluruhnya NaN: dibuang SklImpute
            continue
        if var.is_discrete:
            if len(var.values) < 2:    # dibuang Continuize (continuize.py:21)
                continue
            count += len(var.values) - len(np.unique(present)) + 1
        elif present.min() == present.max():
            count += 1
    return count


def _row(jenis, k, seri, s, a, b, rtol, atol):
    return {"jenis": jenis, "pc": k + 1, "seri": seri, "s": s, "rtol": rtol,
            "atol": atol, "allclose": bool(np.allclose(a, b, rtol=rtol, atol=atol)),
            "maks": float(np.max(np.abs(a - b)))}


def compare(ref_comp, ref_scores, got_comp, got_scores, n_null):
    """Satu baris per komponen per jenis ("komponen"/"skor"), aturan uji 1-3.

    - n_null dianggap jumlah pasti arah nol. zero_variance_count hanya batas bawah
      (aturan uji poin 2); untuk iris dan heart_disease sama dengan jumlah
      explained_variance_ < 1e-10 terukur (diuji). Untuk data baru cek dulu.
    - n_null komponen terakhir (ruang nol): komponen tidak dibandingkan; skor
      dibandingkan dengan atol=1e-8, rtol=0 (nilainya noise ~1e-15, jadi galat
      relatif tidak bermakna).
    - Komponen seri di acuan: s = sign(a_k . b_k), dipakai untuk komponen **dan**
      kolom skor ke-k. Selain itu s = 1.
    - Toleransi lain: np.allclose default (rtol=1e-5, atol=1e-8).
    """
    k_all = len(ref_comp)
    n_nonnull = k_all - n_null
    tied = set(tied_rows(ref_comp[:n_nonnull]))
    rows = []
    for k in range(k_all):
        if k >= n_nonnull:
            rows.append(_row("skor", k, False, 1, ref_scores[:, k],
                             got_scores[:, k], 0, ATOL))
            continue
        seri = k in tied
        s = int(np.sign(ref_comp[k] @ got_comp[k])) if seri else 1
        if s == 0:
            raise ValueError(f"PC{k + 1} seri dan a_k . b_k = 0: tanda s tidak bisa "
                             "ditentukan (komponen pembanding tegak lurus atau nol)")
        rows.append(_row("komponen", k, seri, s, ref_comp[k], s * got_comp[k],
                         RTOL, ATOL))
        rows.append(_row("skor", k, seri, s, ref_scores[:, k],
                         s * got_scores[:, k], RTOL, ATOL))
    return rows


def compare_variance(ref, got, n_null):
    """explained_variance_ atau rasio, dua baris: PC non-nol np.allclose default;
    PC ruang nol (~1e-15, sebagian dipotong ke 0) rtol=0, atol=1e-8, aturan yang sama
    dengan skor ruang nol (aturan uji poin 2)."""
    k = len(ref) - n_null
    rows = []
    for pc, a, b, rtol in [(f"PC1-{k}", ref[:k], got[:k], RTOL),
                           (f"PC{k + 1}-{len(ref)}", ref[k:], got[k:], 0)]:
        if a.size:
            rows.append({"pc": pc, "rtol": rtol, "atol": ATOL,
                         "allclose": bool(np.allclose(a, b, rtol=rtol, atol=ATOL)),
                         "maks": float(np.max(np.abs(a - b)))})
    return rows


def compare_csv(csv, exact):
    """CSV Save Data vs nilai float aslinya (.pkl/.npy). %g = 6 digit signifikan,
    galat murni relatif (<= 5e-6), jadi rtol=1e-5 dan atol=0."""
    diff = np.abs(csv - exact)
    nonzero = exact != 0
    return {"rtol": RTOL, "atol": 0,
            "allclose": bool(np.allclose(csv, exact, rtol=RTOL, atol=0)),
            "maks": float(diff.max()),
            "maks_rel": float((diff[nonzero] / np.abs(exact[nonzero])).max())}


def compare_ratio(csv_ratio, ref_ratio, n_null):
    """Rasio varians dari CSV Components vs acuan, dua baris:
    - PC non-nol: pembulatan %g, sama dengan compare_csv (rtol=1e-5, atol=0).
    - PC ruang nol: rasio ~1e-16 adalah noise numerik yang bisa berbeda antar BLAS,
      jadi aturan skor ruang nol (rtol=0, atol=1e-8)."""
    k = len(ref_ratio) - n_null
    rows = [{"pc": f"PC1-{k}", **compare_csv(csv_ratio[:k], ref_ratio[:k])}]
    if n_null:
        a, b = csv_ratio[k:], ref_ratio[k:]
        rows.append({"pc": f"PC{k + 1}-{len(ref_ratio)}", "rtol": 0, "atol": ATOL,
                     "allclose": bool(np.allclose(a, b, rtol=0, atol=ATOL)),
                     "maks": float(np.max(np.abs(a - b))), "maks_rel": None})
    return rows
