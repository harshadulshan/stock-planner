"""
The M5 data (Walmart unit sales, 3,049 products in 10 stores, 1,969 days) and the official M5 accuracy measure.

Data: the M5 competition files as published by the organisers, from a pinned commit of github.com/Nixtla/m5-forecasts
(datasets/m5.zip), checked by SHA-256 and kept in data_cache/m5. Not bundled with this project.

Accuracy (WRMSSE): the 30,490 product-store series are summed into 12 levels (total, state, store, category, ...,
product, product-store), 42,840 series in all. For each series the squared error of the 28-day forecast is scaled by
the mean squared day-to-day change of its own history (from its first sale on), and the root is weighted by the
series' dollar sales in the last 28 days of the training data (official weights file). The final score is the mean of
the 12 level scores. Uncertainty (SPL): the pinball loss of each quantile scaled by the mean absolute day-to-day
change, weighted the same way.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache", "m5")
URL = ("https://github.com/Nixtla/m5-forecasts/raw/72b8e7fd3b565b3c538adcb1d1a05117d8562d7e/datasets/m5.zip")
SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
FILES = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv", "sales_test_evaluation.csv",
         "weights_evaluation.csv", "sales_train_validation.csv", "sales_test_validation.csv",
         "weights_validation.csv"]
LEVELS = [  # (name, keys) in the order of the official weights file; keys are columns of the id table
    ("Level1", []), ("Level2", ["state_id"]), ("Level3", ["store_id"]), ("Level4", ["cat_id"]),
    ("Level5", ["dept_id"]), ("Level6", ["state_id", "cat_id"]), ("Level7", ["state_id", "dept_id"]),
    ("Level8", ["store_id", "cat_id"]), ("Level9", ["store_id", "dept_id"]), ("Level10", ["item_id"]),
    ("Level11", ["state_id", "item_id"]), ("Level12", ["item_id", "store_id"])]
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]


def fetch():
    """Download and unpack the M5 files once (about 50 MB), checking the archive's SHA-256."""
    if all(os.path.exists(os.path.join(CACHE, f)) for f in FILES):
        return CACHE
    os.makedirs(CACHE, exist_ok=True)
    arc = os.path.join(CACHE, "m5.zip")
    if not os.path.exists(arc):
        with urllib.request.urlopen(URL, timeout=600) as r, open(arc + ".part", "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(arc + ".part", arc)
    h = hashlib.sha256()
    with open(arc, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != SHA256:
        os.remove(arc)
        raise ValueError("m5.zip does not match its checksum")
    with zipfile.ZipFile(arc) as z:                    # regular files only, by base name
        for info in z.infolist():
            base = os.path.basename(info.filename.replace("\\", "/"))
            if not info.is_dir() and base in FILES:
                with z.open(info) as src, open(os.path.join(CACHE, base), "wb") as out:
                    for b in iter(lambda: src.read(1 << 20), b""):
                        out.write(b)
    return CACHE


def load(period="evaluation"):
    """(ids, train, test): ids = item/dept/cat/store/state per series, train = units sold per day before the 28 test
    days (float32, series x days), test = the 28 days to forecast. period "evaluation" is the official final test
    (days 1942-1969, trained on 1-1941); "validation" is the public leaderboard period (1914-1941)."""
    fetch()
    tr = pd.read_csv(os.path.join(CACHE, f"sales_train_{period}.csv"))
    te = pd.read_csv(os.path.join(CACHE, f"sales_test_{period}.csv"))
    ids = tr[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    if not (te[["item_id", "store_id"]].values == ids[["item_id", "store_id"]].values).all():
        raise ValueError("train and test rows are not in the same order")
    dcols = [c for c in tr.columns if c.startswith("d_")]
    tcols = [c for c in te.columns if c.startswith("d_")]
    return ids, tr[dcols].to_numpy(np.float32), te[tcols].to_numpy(np.float32)


def calendar():
    fetch()
    return pd.read_csv(os.path.join(CACHE, "calendar.csv"), parse_dates=["date"])


def prices(ids, days):
    """Shelf price of every series on every day (NaN when the product was not on sale yet), days = 1..days."""
    fetch()
    cal = calendar().iloc[:days]
    sp_ = pd.read_csv(os.path.join(CACHE, "sell_prices.csv"))
    key = ids["item_id"] + "|" + ids["store_id"]
    weeks = cal["wm_yr_wk"].to_numpy()
    uw = np.unique(weeks)
    wk_pos = {w: k for k, w in enumerate(uw)}
    sp_["key"] = sp_["item_id"] + "|" + sp_["store_id"]
    row = pd.Series(np.arange(len(ids)), index=key)
    sp_ = sp_[sp_["key"].isin(row.index) & sp_["wm_yr_wk"].isin(wk_pos)]
    W = np.full((len(ids), len(uw)), np.nan, np.float32)
    W[row[sp_["key"]].to_numpy(), sp_["wm_yr_wk"].map(wk_pos).to_numpy()] = sp_["sell_price"].to_numpy()
    return W[:, [wk_pos[w] for w in weeks]]


# ------------------------------------------------------------------------------------------------ hierarchy
class Hierarchy:
    """The 42,840 series of the 12 levels as a sparse 0/1 matrix over the 30,490 bottom series, in the row order of
    the official weights file, with the weights."""

    def __init__(self, ids, period="evaluation"):
        fetch()
        w = pd.read_csv(os.path.join(CACHE, f"weights_{period}.csv"))
        rows, cols, self.level_of, self.names, wts = [], [], [], [], []
        n = len(ids)
        r = 0
        for lev, keys in LEVELS:
            ww = w[w["Level_id"] == lev]
            if keys:
                k1 = ids[keys[0]].astype(str)
                lab = k1 if len(keys) == 1 else k1 + "|" + ids[keys[1]].astype(str)
                want = ww["Agg_Level_1"].astype(str) if len(keys) == 1 else \
                    ww["Agg_Level_1"].astype(str) + "|" + ww["Agg_Level_2"].astype(str)
            else:
                lab = pd.Series(["Total"] * n)
                want = ww["Agg_Level_1"].astype(str)
            pos = pd.Series(np.arange(len(want)), index=want.to_numpy())
            g = pos.reindex(lab.to_numpy()).to_numpy()
            if np.isnan(g).any():
                raise ValueError(f"{lev}: series without a weight")
            rows.append(r + g.astype(int))
            cols.append(np.arange(n))
            wts.append(ww["weight"].to_numpy(np.float64))
            self.level_of += [lev] * len(want)
            self.names += list(want)
            r += len(want)
        self.A = sp.csr_matrix((np.ones(n * len(LEVELS), np.float32), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(r, n))
        self.weight = np.concatenate(wts)
        self.level_of = np.array(self.level_of)

    def aggregate(self, X):
        return np.asarray(self.A @ X, dtype=np.float64)


def scales(train, squared=True):
    """Mean (squared or absolute) day-to-day change of each series from its first sale on."""
    started = np.maximum.accumulate(train > 0, axis=1)
    d = np.diff(train, axis=1)
    d = d * d if squared else np.abs(d)
    m = started[:, :-1]                      # changes from the first sale on (not the jump up to it)
    s = (d * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)
    return s


def wrmsse(h: Hierarchy, train, test, forecast):
    """Official M5 accuracy: {level: score} and "WRMSSE" = their mean. Arrays are bottom-level series x days."""
    T, Y, F = h.aggregate(train), h.aggregate(test), h.aggregate(forecast)
    rmsse = np.sqrt(((Y - F) ** 2).mean(axis=1) / np.maximum(scales(T), 1e-12))
    out = {}
    for lev, _ in LEVELS:
        m = h.level_of == lev
        out[lev] = float((h.weight[m] * rmsse[m]).sum())
    out["WRMSSE"] = float(np.mean([out[lev] for lev, _ in LEVELS]))
    return out


def spl_bottom(h: Hierarchy, train, test, quantiles: dict):
    """Scaled pinball loss of the product-store level (level 12) for each quantile and their mean, with the official
    level-12 weights: comparable with the level-12 column of the M5 uncertainty results."""
    m = h.level_of == "Level12"
    w = h.weight[m]
    rows = np.asarray(h.A[m].argmax(axis=1)).ravel()         # weights-file order -> bottom series
    sc = np.maximum(scales(train[rows], squared=False), 1e-12)
    Y = test[rows]
    out = {}
    for q, F in quantiles.items():
        F = F[rows]
        e = Y - F
        pin = np.where(e >= 0, q * e, (q - 1) * e).mean(axis=1)
        out[q] = float((w * pin / sc).sum())
    out["mean"] = float(np.mean([out[q] for q in quantiles]))
    return out


def snaive(train, horizon=28):
    return np.tile(train[:, -7:], (1, horizon // 7 + 1))[:, :horizon]


def naive(train, horizon=28):
    return np.repeat(train[:, -1:], horizon, axis=1)


def read_submission(path, ids, period="evaluation"):
    """A Kaggle M5 submission file (id, F1..F28) as series x days in the order of ids."""
    sub = pd.read_csv(path)
    sub = sub[sub["id"].str.endswith(period)].set_index("id")
    key = ids["item_id"] + "_" + ids["store_id"] + "_" + period
    return sub.loc[key.to_numpy(), [f"F{k}" for k in range(1, 29)]].to_numpy(np.float64)
