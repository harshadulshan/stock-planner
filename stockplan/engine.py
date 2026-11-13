"""
The forecast engine: how many units of each product will sell on each of the next days.

One gradient-boosted tree model (LightGBM) learns from all products at once, the way the leading M5 entries did:
for a day t it sees what sold up to `horizon` days before t (recent days, weekly averages, the same weekday in past
weeks, how long since the last sale), the calendar (weekday, month, holidays and events), the price when known
(level, change, discount against the usual price) and the product's group (category, department, supplier). Because
every input is at least `horizon` days old, one model forecasts all days up to `horizon` ahead at once, and the same
model can be reused on later days without retraining: its inputs never look past the day of the decision.

Errors are modelled as negative binomial around the forecast (see uncertainty.py): the forecast is the mean, the
range comes from how far actual sales strayed from forecasts in a held-out period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

ROLL = (7, 28, 56, 112, 364)
DEFAULT_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.05, num_leaves=127,
                      min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                      lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=2, seed=7, deterministic=True,
                      force_row_wise=True)


@dataclass
class History:
    """Units sold per product and day, and what is known about the days ahead.

    Y       (n, T) units sold per day (days before a product was first sold may be 0)
    start   date of column 0
    static  n rows of product groups (categorical columns, e.g. category, supplier, department, item)
    price   (n, T + ahead) shelf price per day, also for the days ahead (NaN = not on sale), or None
    events  (T + ahead) rows with columns "event" and "event_type" (text or empty), or None
    flags   (n, T + ahead) extra 0/1 calendar flags per product (e.g. food-stamp days in its state), or None
    """
    Y: np.ndarray
    start: pd.Timestamp
    static: pd.DataFrame
    price: np.ndarray | None = None
    events: pd.DataFrame | None = None
    flags: np.ndarray | None = None

    @property
    def n(self):
        return self.Y.shape[0]

    @property
    def T(self):
        return self.Y.shape[1]

    def dates(self, days):
        return pd.DatetimeIndex(self.start + pd.to_timedelta(np.asarray(days), unit="D"))


@dataclass
class Model:
    booster: lgb.Booster
    horizon: int
    columns: list
    cats: list
    codes: dict = field(default_factory=dict)      # category -> {value: code}
    rounds: int = 0


# ------------------------------------------------------------------------------------------------ features
class _Stats:
    """Running sums over the history, so that any window mean is two lookups."""

    def __init__(self, Y):
        y = np.nan_to_num(Y.astype(np.float64))
        n, T = y.shape
        self.y = y
        self.c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(y, axis=1)], axis=1)
        self.c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(y * y, axis=1)], axis=1)
        sold = y > 0
        self.first = np.where(sold.any(axis=1), sold.argmax(axis=1), T)        # first day with a sale
        idx = np.where(sold, np.arange(T)[None, :], -1)
        self.last = np.maximum.accumulate(idx, axis=1)                          # last sale up to each day

    def window(self, u, w):
        """Mean and std of the w days ending at day u (vector of days, one per row or broadcast)."""
        a = np.clip(u + 1 - w, 0, None)
        b = np.clip(u + 1, 0, None)
        rows = np.arange(self.y.shape[0])[:, None]
        cnt = np.maximum(b - a, 1)
        s1 = self.c1[rows, b] - self.c1[rows, a]
        s2 = self.c2[rows, b] - self.c2[rows, a]
        m = s1 / cnt
        return m, np.sqrt(np.maximum(s2 / cnt - m * m, 0))


def _codes(values, known=None):
    v = pd.Series(values).fillna("").astype(str)
    if known is None:
        known = {k: i for i, k in enumerate(sorted(v.unique()))}
    return v.map(known).fillna(-1).to_numpy(np.int32), known


def features(h: History, horizon: int, days, stats: _Stats | None = None, codes=None):
    """Feature table for every product on each of `days` (day numbers, 0 = h.start), using sales up to
    day - horizon only. Returns (columns dict of (n, len(days)) arrays, categorical column names, codes)."""
    days = np.asarray(days)
    if days.max() - horizon > h.T - 1:
        raise ValueError("a day is too far ahead for this horizon")
    st = stats or _Stats(h.Y)
    n, D = h.n, len(days)
    T = h.T
    u = np.broadcast_to(days[None, :] - horizon, (n, D))                         # last day of sales seen
    rows = np.arange(n)[:, None]
    f = {}
    for k in range(7):
        t = np.clip(days - horizon - k, 0, T - 1)
        f[f"lag_{k}"] = np.where(days - horizon - k >= 0, st.y[:, t], np.nan)
    m0 = -(-horizon // 7)                                                       # same weekday, at least horizon back
    same = [np.clip(days - 7 * (m0 + j), 0, T - 1) for j in range(8)]
    vals = np.stack([st.y[:, s] for s in same])
    f["same_weekday_4"] = vals[:4].mean(axis=0)
    f["same_weekday_8"] = vals.mean(axis=0)
    for w in ROLL:
        m, s = st.window(u, w)
        f[f"mean_{w}"] = m
        if w in (7, 28):
            f[f"std_{w}"] = s
    uu = np.clip(u, 0, T - 1)
    last = st.last[rows, uu]
    f["days_since_sale"] = np.where(last >= 0, u - last, np.minimum(u + 1, 999)).astype(np.float64)
    f["age"] = (u - st.first[:, None]).astype(np.float64)
    m28, _ = st.window(u, 28)
    m112, _ = st.window(u, 112)
    f["trend"] = (m28 + 0.1) / (m112 + 0.1)
    # last year: the same week, the four weeks starting on the same date, and the growth since
    dd = np.broadcast_to(days[None, :], (n, D))
    have = dd - 367 >= 0
    ly, _ = st.window(dd - 361, 7)
    ly_next, _ = st.window(dd - 337, 28)
    m28_ly, _ = st.window(u - 364, 28)
    f["last_year_week"] = np.where(have, ly, np.nan)
    f["last_year_next_4w"] = np.where(have, ly_next, np.nan)
    f["growth_vs_last_year"] = np.where(u - 364 - 28 >= 0, (m28 + 0.1) / (m28_ly + 0.1), np.nan)
    # calendar
    dates = h.dates(days)
    for name, val in (("weekday", dates.weekday), ("day_of_month", dates.day), ("month", dates.month),
                      ("week_of_year", dates.isocalendar().week.to_numpy())):
        f[name] = np.broadcast_to(np.asarray(val, np.float64)[None, :], (n, D))
    cats = []
    codes = dict(codes or {})
    if h.events is not None:
        for col in ("event", "event_type"):
            c, codes[col] = _codes(h.events[col].to_numpy()[days], codes.get(col))
            f[col] = np.broadcast_to(c[None, :].astype(np.float64), (n, D))
            cats.append(col)
        ev = h.events["event"].fillna("").astype(str).to_numpy() != ""
        evi = np.flatnonzero(ev)
        nxt = np.searchsorted(evi, days)
        dist = np.where(nxt < len(evi), evi[np.minimum(nxt, len(evi) - 1)] - days, 99)
        f["days_to_event"] = np.broadcast_to(np.minimum(dist, 30).astype(np.float64)[None, :], (n, D))
    if h.flags is not None:
        f["flag"] = h.flags[:, days].astype(np.float64)
    if h.price is not None:
        p = h.price.astype(np.float64)
        pr = p[:, days]
        known = np.where(np.isnan(p[:, :T]), -np.inf, p[:, :T])
        top = np.max(known, axis=1)
        top = np.where(np.isfinite(top), top, np.nanmax(p, axis=1))
        f["price"] = pr
        f["price_vs_max"] = pr / top[:, None]
        f["price_vs_mean"] = pr / np.nanmean(p, axis=1)[:, None]
        prev = p[:, np.clip(days - 7, 0, None)]
        f["price_change"] = pr / prev - 1
    for col in h.static.columns:
        c, codes[col] = _codes(h.static[col].to_numpy(), codes.get(col))
        f[col] = np.broadcast_to(c[:, None].astype(np.float64), (n, D))
        cats.append(col)
    return f, cats, codes


def _usable(h: History, days, st: _Stats):
    """Rows to learn from: the product had been sold before (or was on sale, when prices are known)."""
    ok = days[None, :] >= st.first[:, None]
    if h.price is not None:
        ok &= ~np.isnan(h.price[:, days])
    return ok


def _table(f, order):
    return np.column_stack([np.asarray(f[c], np.float32).ravel() for c in order])


def fit(h: History, horizon: int, first_day: int | None = None, rounds: int = 600, params=None, last_day=None,
        weight_recent=0.0, log=None) -> Model:
    """Learn from days first_day..last_day (default: the last two years up to the end of the history)."""
    last_day = h.T - 1 if last_day is None else last_day
    first_day = max(horizon + 7, last_day - 730 if first_day is None else first_day)
    days = np.arange(first_day, last_day + 1)
    st = _Stats(h.Y[:, :last_day + 1]) if last_day < h.T - 1 else _Stats(h.Y)
    hh = h if last_day == h.T - 1 else History(h.Y[:, :last_day + 1], h.start, h.static, h.price, h.events, h.flags)
    f, cats, codes = features(hh, horizon, days, st)
    ok = _usable(hh, days, st)
    order = list(f.keys())
    X = _table(f, order)[ok.ravel()]
    y = hh.Y[:, days].astype(np.float32).ravel()[ok.ravel()]
    del f
    w = None
    if weight_recent:
        age = (last_day - np.broadcast_to(days[None, :], ok.shape))[ok]
        w = (1 + weight_recent * (age < 365)).astype(np.float32)
    ds = lgb.Dataset(X, y, weight=w, feature_name=order, categorical_feature=[order.index(c) for c in cats],
                     free_raw_data=True)
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    if len(y) < 300_000:                            # small shops: smaller trees
        p.update(num_leaves=min(p["num_leaves"], 31), min_data_in_leaf=min(p["min_data_in_leaf"], 20))
    cb = [lgb.log_evaluation(0)]
    bst = lgb.train(p, ds, num_boost_round=rounds, callbacks=cb)
    if log:
        log(f"trained on {len(y):,} rows")
    return Model(bst, horizon, order, cats, codes, rounds)


def predict(model: Model, h: History, days, rounds=None):
    """Mean units sold per product on each of `days` (n, len(days)); never negative."""
    days = np.asarray(days)
    f, _, _ = features(h, model.horizon, days, codes=model.codes)
    X = _table(f, model.columns)
    pred = model.booster.predict(X, num_iteration=rounds or model.rounds)
    out = np.maximum(pred.reshape(h.n, len(days)), 0)
    if h.price is not None:                           # not on sale: nothing sells
        out = np.where(np.isnan(h.price[:, days]), 0, out)
    return out
