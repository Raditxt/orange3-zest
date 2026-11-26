"""Bukti 7 klaim tentang cara Orange 3.40.0 membangun Tree (pembanding ExtDT).

Jalankan dengan Python bawaan Orange:
    D:\\Orange\\python.exe docs\\proofs\\tree_internals_check.py

Setiap klaim punya satu fungsi check_klaimN() yang mencetak angka dan memakai
assert, jadi exit code 0 berarti ketujuh klaim terbukti. Spesifikasinya ada di
docs/orange-tree-internals.md. Data: iris (klasifikasi, numerik murni) dan
heart_disease (klasifikasi campuran diskret/numerik dengan NaN; juga dipakai
untuk regresi dengan target "max HR").
"""
import itertools

import numpy as np

import Orange
from Orange.data import Table, Domain, DiscreteVariable
from Orange.modelling.tree import TreeLearner
from Orange.classification.tree import TreeLearner as ClsTreeLearner
from Orange.regression.tree import TreeLearner as RegTreeLearner
from Orange.tree import Node, DiscreteNode, MappedDiscreteNode, NumericNode, TreeModel
from Orange.widgets.model.owtree import OWTreeLearner

IRIS = Table("iris")
HD = Table("heart_disease")


def _widget_kwargs(**ubah):
    """learner_kwargs() widget Tree tanpa membuka GUI.

    learner_kwargs() hanya membaca atribut Setting dan self.preprocessors,
    jadi cukup objek tiruan berisi nilai default Setting (bisa diubah).
    """
    class Palsu:
        preprocessors = None          # tidak ada sinyal Preprocessor masuk
    for nama in ("binary_trees", "limit_min_leaf", "min_leaf",
                 "limit_min_internal", "min_internal", "limit_depth",
                 "max_depth", "limit_majority", "sufficient_majority"):
        setattr(Palsu, nama, getattr(OWTreeLearner, nama).default)
    for nama, nilai in ubah.items():
        setattr(Palsu, nama, nilai)
    return OWTreeLearner.learner_kwargs(Palsu())


def _hd_regresi():
    """heart_disease dengan target numerik "max HR" (atribut lain tetap)."""
    t = HD.domain["max HR"]
    dom = Domain([a for a in HD.domain.attributes if a is not t], t)
    return HD.transform(dom)


def _nodes(node):
    """Semua node (bukan None) secara pre-order."""
    if node is None:
        return
    yield node
    for c in node.children:
        yield from _nodes(c)


# ---------- skor brute-force NumPy (pembanding independen) ----------

def _entropy(y, n_cls):
    p = np.bincount(y.astype(int), minlength=n_cls) / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _gain_cls(y, groups, n_cls):
    """Information gain (bit) dari pembagian y ke label groups."""
    h = _entropy(y, n_cls)
    for g in np.unique(groups):
        m = groups == g
        h -= m.mean() * _entropy(y[m], n_cls)
    return h


def _gain_reg(y, groups):
    """Penurunan MSE: var(y) - rata-rata berbobot var anak (ddof=0)."""
    v = np.var(y)
    for g in np.unique(groups):
        m = groups == g
        v -= m.mean() * np.var(y[m])
    return v


def _brute_root(data, binarize, min_leaf, regresi=False):
    """Skor terbaik tiap atribut di akar, dihitung ulang dengan NumPy.

    Return list (skor, deskripsi split) per atribut, urut domain.
    """
    y_all = data.Y
    n_cls = 0 if regresi else len(data.domain.class_var.values)
    gain = (lambda y, g: _gain_reg(y, g)) if regresi else \
        (lambda y, g: _gain_cls(y, g, n_cls))
    hasil = []
    for j, attr in enumerate(data.domain.attributes):
        x = data.X[:, j]
        known = ~np.isnan(x)
        xk, yk, frac = x[known], y_all[known], known.mean()
        best = (0.0, None)
        if attr.is_continuous:
            for t in np.unique(xk)[:-1]:          # kandidat = nilai data
                left = xk <= t
                if left.sum() < min_leaf or (~left).sum() < min_leaf:
                    continue
                s = gain(yk, left.astype(int)) * frac
                if s > best[0] + 1e-12:
                    best = (s, t)
        elif binarize and len(attr.values) > 2:
            nv = len(attr.values)
            for mask in range(1, 2 ** (nv - 1)):  # semua 2-grup nontrivial
                side = (mask >> xk.astype(int)) & 1
                if side.sum() < min_leaf or (1 - side).sum() < min_leaf:
                    continue
                s = gain(yk, side) * frac
                if s > best[0] + 1e-12:
                    best = (s, mask)
        else:                                     # multiway, satu cabang/nilai
            counts = np.bincount(xk.astype(int), minlength=len(attr.values))
            ok = counts >= min_leaf                # cabang kecil dibuang
            if ok.sum() >= 2:
                m = ok[xk.astype(int)]
                s = gain(yk[m], xk[m]) * m.sum() / len(x)
                best = (s, "multiway")
        hasil.append(best)
    return hasil


# ------------------------------- klaim -------------------------------

def check_klaim1():
    print("=== Klaim 1: widget Tree memakai TreeLearner Orange, bukan sklearn ===")
    print("Orange versi:", Orange.version.version)
    assert Orange.version.version.startswith("3.40.0")

    fitter_cls = OWTreeLearner.LEARNER
    print("OWTreeLearner.LEARNER :", fitter_cls.__module__ + "." + fitter_cls.__name__)
    assert fitter_cls is TreeLearner
    print("__fits__              :", {k: v.__module__ + "." + v.__name__
                                       for k, v in fitter_cls.__fits__.items()})
    assert fitter_cls.__fits__["classification"] is ClsTreeLearner
    assert fitter_cls.__fits__["regression"] is RegTreeLearner

    kw = _widget_kwargs()
    print("learner_kwargs default:", kw)
    assert kw == dict(max_depth=100, min_samples_split=5, min_samples_leaf=2,
                      binarize=True, preprocessors=None, sufficient_majority=0.95)

    f = fitter_cls(**kw)
    cls_l, reg_l = f.get_learner("classification"), f.get_learner("regression")
    print("klasifikasi params    :", cls_l.params)
    print("regresi params        :", reg_l.params)
    assert type(cls_l) is ClsTreeLearner and type(reg_l) is RegTreeLearner
    assert "sufficient_majority" not in reg_l.params
    print("preprocessor bawaan   : klasifikasi", cls_l.preprocessors,
          "| regresi", reg_l.preprocessors)
    assert tuple(cls_l.preprocessors) == () and tuple(reg_l.preprocessors) == ()

    m_iris = f(IRIS)
    m_reg = f(_hd_regresi())
    print("model iris / regresi  :", type(m_iris).__name__, "/", type(m_reg).__name__)
    assert type(m_iris) is TreeModel and type(m_reg) is TreeModel

    # Preprocessor dari sinyal widget hanya sampai ke learner klasifikasi:
    # Fitter menyaring kwargs berdasarkan co_varnames __init__ learner, dan
    # __init__ regresi tidak punya nama "preprocessors".
    f2 = fitter_cls(**{**kw, "preprocessors": ["PENANDA"]})
    pc = f2.get_learner("classification").preprocessors
    pr = f2.get_learner("regression").preprocessors
    print("preprocessor sinyal   : klasifikasi", pc, "| regresi", pr)
    assert tuple(pc) == ("PENANDA",) and tuple(pr) == ()


def check_klaim2():
    print("=== Klaim 2: kriteria split = information gain (klasifikasi), "
          "penurunan MSE (regresi) ===")
    kw = _widget_kwargs()
    for nama, data, regresi in (("iris", IRIS, False),
                                ("heart_disease", HD, False),
                                ("heart_disease reg (max HR)", _hd_regresi(), True)):
        model = TreeLearner(**kw)(data)
        skor = _brute_root(data, True, kw["min_samples_leaf"], regresi)
        s = np.array([b[0] for b in skor])
        pemenang = int(np.flatnonzero(s >= s.max() - 1e-12)[0])  # seri -> indeks pertama
        print(f"{nama}: akar Orange = {model.root.attr.name!r}, "
              f"brute-force = {data.domain.attributes[pemenang].name!r}, "
              f"skor = {s[pemenang]:.10f}")
        assert model.root.attr_idx == pemenang
        if not regresi and nama == "iris":
            seri = [data.domain.attributes[i].name
                    for i in np.flatnonzero(np.isclose(s, s.max(), rtol=0, atol=1e-12))]
            print("  skor seri di akar iris:", seri, "-> yang dipilih indeks pertama")
            assert seri == ["petal length", "petal width"]

    # Skor scorer Cython = rumus NumPy (bukan Gini, bukan gain ratio)
    from Orange.classification import _tree_scorers as ts
    x = np.ascontiguousarray(IRIS.X[:, 0])
    sc, cut = ts.find_threshold_entropy(x, IRIS.Y, np.argsort(x), 3, 2)
    ig = _gain_cls(IRIS.Y, (x > cut).astype(int), 3)
    print(f"iris sepal length: Cython {sc:.12f} vs IG NumPy {ig:.12f} "
          f"(selisih {abs(sc - ig):.1e})")
    assert np.isclose(sc, ig, rtol=0, atol=1e-10)
    r = _hd_regresi()
    j = [a.name for a in r.domain.attributes].index("age")
    x = np.ascontiguousarray(r.X[:, j])
    sc, cut = ts.find_threshold_MSE(x, r.Y, np.argsort(x), 2)
    dm = _gain_reg(r.Y, (x > cut).astype(int))
    print(f"regresi age      : Cython {sc:.10f} vs dMSE NumPy {dm:.10f} "
          f"(selisih {abs(sc - dm):.1e})")
    assert np.isclose(sc, dm, rtol=0, atol=1e-8)


def check_klaim3():
    print("=== Klaim 3: atribut diskret = multiway (binarize=False) atau "
          "2-grup optimal (binarize=True) ===")
    base = _widget_kwargs()

    m_multi = TreeLearner(**{**base, "binarize": False})(HD)
    m_bin = TreeLearner(**base)(HD)
    r = m_multi.root
    print(f"binarize=False: akar {type(r).__name__} {r.attr.name!r}, "
          f"{len(r.children)} cabang, nilai = {r.attr.values}")
    assert type(r) is DiscreteNode and len(r.children) == len(r.attr.values)
    r = m_bin.root
    print(f"binarize=True : akar {type(r).__name__} {r.attr.name!r}, "
          f"{len(r.children)} cabang, mapping nilai->cabang = {r.mapping.tolist()}")
    assert type(r) is MappedDiscreteNode and len(r.children) == 2

    # mapping = pengelompokan terbaik dari brute-force semua 2^(k-1)-1 kelompok
    skor = _brute_root(HD, True, base["min_samples_leaf"])
    _, mask = skor[r.attr_idx]
    mapping_bf = [(mask >> v) & 1 for v in range(len(r.attr.values))]
    print("  mapping brute-force:", mapping_bf)
    assert r.mapping.tolist() == mapping_bf

    # Atribut biner tetap DiscreteNode walau binarize=True
    tipe = {(type(n).__name__, len(n.attr.values))
            for n in _nodes(m_bin.root) if n.attr is not None and n.attr.is_discrete}
    print("  (tipe node, jumlah nilai) diskret di pohon binarize=True:", sorted(tipe))
    assert all(t == "DiscreteNode" for t, k in tipe if k == 2)
    assert all(t == "MappedDiscreteNode" for t, k in tipe if k > 2)

    # Batas binarisasi exhaustive: > 16 nilai ditolak
    print("  MAX_BINARIZATION =", ClsTreeLearner.MAX_BINARIZATION)
    banyak = DiscreteVariable("v17", values=tuple(f"n{i}" for i in range(17)))
    dom = Domain([banyak], IRIS.domain.class_var)
    data = Table.from_numpy(dom, (np.arange(150) % 17)[:, None].astype(float), IRIS.Y)
    try:
        TreeLearner(**base)(data)
        raise AssertionError("seharusnya ValueError")
    except ValueError as e:
        print("  17 nilai + binarize=True -> ValueError:", e)
    TreeLearner(**{**base, "binarize": False})(data)   # multiway tetap jalan


def check_klaim4():
    print("=== Klaim 4: threshold numerik = nilai data (maks sisi kiri), "
          "bukan titik tengah ===")
    kw = _widget_kwargs()
    total = 0
    for nama, data in (("iris", IRIS), ("heart_disease", HD),
                       ("heart_disease reg", _hd_regresi())):
        model = TreeLearner(**kw)(data)
        for n in _nodes(model.root):
            if not isinstance(n, NumericNode):
                continue
            col = data.X[n.subset, n.attr_idx]
            col = col[~np.isnan(col)]
            kiri = col[col <= n.threshold]
            assert n.threshold in col                  # nilai yang ada di data
            assert n.threshold == kiri.max()           # = maks sisi kiri
            total += 1
    print(f"{total} NumericNode (iris + heart_disease klas./reg.): "
          "threshold selalu nilai data = maks sisi kiri")

    # Arah: x <= t ke cabang 0, x > t ke cabang 1
    model = TreeLearner(**kw)(IRIS)
    r = model.root
    arah = [r.descend(np.array([0, 0, v, 0])) for v in (1.9, 1.9001)]
    print(f"iris akar: {r.attr.name} threshold {r.threshold}; "
          f"x=1.9 -> cabang {arah[0]}, x=1.9001 -> cabang {arah[1]}")
    assert arah == [0, 1]
    assert [c.description for c in r.children] == ["≤ 1.9", "> 1.9"]

    # Pembanding sklearn: partisi sama, threshold titik tengah
    from Orange.classification import SklTreeLearner
    skl = SklTreeLearner(criterion="entropy")(IRIS).skl_model
    print(f"sklearn (entropy) akar: fitur {skl.tree_.feature[0]} "
          f"threshold {skl.tree_.threshold[0]:.2f} = (1.9 + 3.0) / 2")
    assert np.isclose(skl.tree_.threshold[0], (1.9 + 3.0) / 2)


def check_klaim5():
    print("=== Klaim 5: missing value ===")
    kw = _widget_kwargs()

    # (a) Baris dengan target NaN tidak dipakai sama sekali
    d = IRIS.copy()
    with d.unlocked():
        d.Y[[0, 60, 120]] = np.nan
    m = TreeLearner(**kw)(d)
    print("(a) target NaN di 3 baris -> akar.subset:", len(m.root.subset), "dari", len(d))
    assert len(m.root.subset) == 147 and not {0, 60, 120} & set(m.root.subset)

    # (b) Tanpa imputasi: baris NaN di atribut split tidak masuk anak mana pun
    m = TreeLearner(**kw)(HD)
    r = m.root
    n_nan = int(np.isnan(HD.X[r.subset, r.attr_idx]).sum())
    anak = sum(len(c.subset) for c in r.children if c is not None)
    print(f"(b) heart_disease akar {r.attr.name!r}: {len(r.subset)} baris, "
          f"{n_nan} NaN, anak menampung {anak}")
    assert n_nan > 0 and anak == len(r.subset) - n_nan

    # (c) Skor dikalikan fraksi baris yang diketahui: 60% NaN di petal length
    #     (skor seri dengan petal width) membuat akar pindah ke petal width
    d = IRIS.copy()
    rng = np.random.default_rng(0)
    with d.unlocked():
        d.X[rng.choice(150, 90, replace=False), 2] = np.nan
    m = TreeLearner(**kw)(d)
    print("(c) 60% NaN di petal length -> akar pindah ke", repr(m.root.attr.name))
    assert m.root.attr.name == "petal width"

    # (d) Prediksi: NaN di atribut split -> berhenti di node itu,
    #     hasilnya distribusi kelas node internal tersebut
    m = TreeLearner(**kw)(IRIS)
    x = IRIS[:1].copy()
    with x.unlocked():
        x.X[0, m.root.attr_idx] = np.nan
    p = m(x, ret=m.Probs)[0]
    print("(d) iris baris 0, petal length NaN -> prob", p.round(4),
          "= akar", m.root.value)
    assert np.allclose(p, m.root.value / m.root.value.sum(), rtol=0, atol=1e-12)


def check_klaim6():
    print("=== Klaim 6: parameter stop ===")
    kw = _widget_kwargs()
    for nama, data, regresi in (("iris", IRIS, False), ("heart_disease", HD, False),
                                ("heart_disease reg", _hd_regresi(), True)):
        m = TreeLearner(**kw)(data)
        semua = list(_nodes(m.root))
        internal = [n for n in semua if n.children]
        min_node = min(len(n.subset) for n in semua)
        min_int = min(len(n.subset) for n in internal)
        print(f"{nama}: {len(semua)} node, depth {m.depth()}, "
              f"min |node| = {min_node} (>= min_samples_leaf 2), "
              f"min |internal| = {min_int} (>= min_samples_split 5)")
        assert min_node >= kw["min_samples_leaf"]
        assert min_int >= kw["min_samples_split"]
        if not regresi:
            maj = max(np.max(np.asarray(n.value)) / np.sum(n.value) for n in internal)
            print(f"  mayoritas maks di node internal = {maj:.4f} (< 0.95)")
            assert maj < kw["sufficient_majority"]

    # max_depth = jumlah level split; depth() model <= max_depth
    for md in (1, 2, 3):
        m = TreeLearner(**{**kw, "max_depth": md})(HD)
        print(f"max_depth={md} -> depth {m.depth()}")
        assert m.depth() == md

    # Checkbox mati di widget -> nilai "tanpa batas"
    off = _widget_kwargs(limit_min_leaf=False, limit_min_internal=False,
                         limit_depth=False, limit_majority=False)
    print("semua checkbox mati:", {k: off[k] for k in
          ("max_depth", "min_samples_split", "min_samples_leaf", "sufficient_majority")})
    assert (off["max_depth"], off["min_samples_split"], off["min_samples_leaf"],
            off["sufficient_majority"]) == (None, 2, 1, 1)
    # sufficient_majority=1: hanya node murni yang berhenti karena mayoritas
    m = TreeLearner(**off)(IRIS)
    murni = all(np.max(np.asarray(n.value)) == np.sum(n.value) for n in _nodes(m.root) if not n.children)
    print("  iris tanpa batas: semua daun murni =", murni, "| depth", m.depth())
    assert murni


def check_klaim7():
    print("=== Klaim 7: bentuk model dan output ===")
    kw = _widget_kwargs()
    m = TreeLearner(**kw)(IRIS)
    tipe = sorted({type(n).__name__ for n in _nodes(m.root)})
    print("iris: tipe node =", tipe, "| node", m.node_count(),
          "| daun", m.leaf_count(), "| depth", m.depth())
    assert set(tipe) <= {"Node", "DiscreteNode", "MappedDiscreteNode", "NumericNode"}
    print("  akar.value (jumlah per kelas, bukan proporsi):", m.root.value)
    assert np.array_equal(m.root.value, [50, 50, 50])

    # Probabilitas = frekuensi relatif daun, tanpa smoothing; kelas = argmax
    for nama, data in (("iris", IRIS), ("heart_disease (ada NaN)", HD)):
        m = TreeLearner(**kw)(data)
        counts = m.get_values_by_nodes(data.X)        # traversal Python murni
        p_ref = counts / counts.sum(axis=1, keepdims=True)
        p = m(data, ret=m.Probs)
        y = m(data)
        d = np.abs(p - p_ref).max()
        print(f"{nama}: prob model vs value daun/jumlah, selisih maks {d:.1e}; "
              f"kelas = argmax: {np.array_equal(y, p.argmax(axis=1))}")
        assert np.allclose(p, p_ref, rtol=0, atol=1e-12)
        assert np.array_equal(y, p.argmax(axis=1))
        assert np.array_equal(m.get_values(data.X), counts)   # kode terkompilasi

    # Regresi: value = [mean, var ddof=0], prediksi = mean daun
    r = _hd_regresi()
    m = TreeLearner(**kw)(r)
    v = m.root.value
    print(f"regresi akar.value = {v.round(4)} vs mean {r.Y.mean():.4f}, "
          f"var(ddof=0) {np.var(r.Y):.4f}")
    assert np.allclose(v, [r.Y.mean(), np.var(r.Y)], rtol=0, atol=1e-9)
    pred = m(r)
    ref = m.get_values_by_nodes(r.X)[:, 0]
    print(f"  prediksi = mean daun, selisih maks {np.abs(pred - ref).max():.1e}")
    assert np.allclose(pred, ref, rtol=0, atol=1e-12)


if __name__ == "__main__":
    for f in (check_klaim1, check_klaim2, check_klaim3, check_klaim4,
              check_klaim5, check_klaim6, check_klaim7):
        f()
        print()
    print("SEMUA KLAIM TERBUKTI")
