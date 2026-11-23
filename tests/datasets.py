"""Dataset uji: bawaan Orange, atau bank-full (UCI) dari luar repo."""
import os
from pathlib import Path

import pytest
from Orange.data import Domain, Table

# bank-full tidak disimpan di repo (docs/orange-pca-internals.md, batas cakupan);
# default = salinan di repo magang-datamining, bisa diganti lewat env ZEST_BANK_FULL.
BANK_FULL = Path(os.environ.get(
    "ZEST_BANK_FULL",
    Path(__file__).resolve().parents[2] / "magang-datamining" / "datasets" / "bank-full.csv"))


def load_table(name):
    """Dataset bawaan Orange, atau "bank-full" dari BANK_FULL dengan `y` sebagai
    kelas (target tidak ikut PCA). Skip bila file bank-full tidak ada."""
    if name != "bank-full":
        return Table(name)
    if not BANK_FULL.exists():
        pytest.skip(f"bank-full tidak ditemukan: {BANK_FULL}")
    raw = Table(str(BANK_FULL))
    y = raw.domain["y"]
    return raw.transform(Domain([a for a in raw.domain.attributes if a is not y], y))
