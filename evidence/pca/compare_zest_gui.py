"""Bukti GUI T2a: output widget PCA Orange vs Zest PCA yang disimpan dari GUI.

Untuk tiap setting, bandingkan file di evidence/pca/:
  heart_pca_<setting>_scores.pkl      vs heart_zestpca_<setting>_scores.pkl      (Save Data)
  heart_pca_<setting>_components.npy  vs heart_zestpca_<setting>_components.npy  (Python Script)
Acuan heart_pca_* = bukti GUI PCA Orange 20 Nov 2026 (setting sama). Dibandingkan
dengan compare() (aturan uji docs/orange-pca-internals.md, rtol=1e-5 atol=1e-8).

Jalankan dari root repo:  D:\\Orange\\python.exe evidence\\pca\\compare_zest_gui.py
Keluar 0 bila semua identik dalam toleransi spec, 1 bila ada yang beda/hilang.
"""
import sys
from pathlib import Path

import numpy as np
from Orange.data import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from pca_compare import compare, zero_variance_count  # noqa: E402

EVIDENCE = ROOT / "evidence" / "pca"
DATA, DATASET = "heart", "heart_disease"
SETTINGS = ["norm-on_25pc"]


def load(widget, setting):
    stem = f"{DATA}_{widget}_{setting}"
    scores = Table(str(EVIDENCE / f"{stem}_scores.pkl")).X
    comps = np.load(EVIDENCE / f"{stem}_components.npy")
    return comps, scores


def main():
    n_null = zero_variance_count(Table(DATASET))
    ok = True
    for setting in SETTINGS:
        try:
            ref_c, ref_s = load("pca", setting)
            got_c, got_s = load("zestpca", setting)
        except FileNotFoundError as err:
            print(f"[{setting}] FILE HILANG: {err.filename}")
            ok = False
            continue
        if ref_c.shape != got_c.shape or ref_s.shape != got_s.shape:
            print(f"[{setting}] BENTUK BEDA: komponen {ref_c.shape} vs {got_c.shape}, "
                  f"skor {ref_s.shape} vs {got_s.shape}")
            ok = False
            continue
        rows = compare(ref_c, ref_s, got_c, got_s, n_null)
        bad = [r for r in rows if not r["allclose"]]
        worst = max(r["maks"] for r in rows)
        status = "IDENTIK" if not bad else "BEDA"
        print(f"[{setting}] {status}: komponen {ref_c.shape}, skor {ref_s.shape}, "
              f"selisih maks {worst:.3g}")
        for r in bad:
            print(f"    {r['jenis']} PC{r['pc']}: maks {r['maks']:.3g}")
        ok = ok and not bad
    print("SEMUA IDENTIK dalam toleransi spec" if ok else "ADA YANG GAGAL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
