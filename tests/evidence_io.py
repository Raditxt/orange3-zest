"""Baca bukti GUI Orange 3.40.0 di evidence/pca/ (lihat evidence/pca/README.md)."""
import csv
from pathlib import Path

import numpy as np
from Orange.data import Table

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence" / "pca"


def _read_csv(name):
    """Header + baris. CSV disimpan Save Data tanpa baris anotasi tipe."""
    with open(EVIDENCE / name, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def read_scores_csv(name):
    """Kolom PC1..PCk dari output Transformed Data; kolom kelas dibuang."""
    header, rows = _read_csv(name)
    cols = [j for j, h in enumerate(header) if h.startswith("PC")]
    return np.array([[float(r[j]) for j in cols] for r in rows])


def read_components_csv(name):
    """Baris PC x fitur dari output Components; kolom `components` dan `variance`
    (rasio varians) dibuang."""
    header, rows = _read_csv(name)
    assert header[:2] == ["components", "variance"], header[:2]
    return np.array([[float(v) for v in r[2:]] for r in rows])


def read_scores_pkl(name):
    """X dari Table yang di-pickle Save Data (output Transformed Data)."""
    return Table(str(EVIDENCE / name)).X


def read_components_npy(name):
    """components_ (k x p) yang ditulis Python Script: np.save(..., in_data.X)."""
    return np.load(EVIDENCE / name)
