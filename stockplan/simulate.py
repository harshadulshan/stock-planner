"""
Replay: run a way of ordering through real (or simulated) days, product by product, and count what happens.

Each day: orders that are due arrive, customers buy what is on the shelf (what they cannot get is a lost sale), and on
review days a new order is placed that arrives `lead` days later. Two ways of ordering are compared:

* the planner: every unit whose chance of selling before the next delivery is worth the money it ties up, from
  the forecast and its range (policy.py), using only sales up to the day before the decision;
* a rule of thumb: keep between `low` and `high` days of average sales (the last 28 days) in stock, the way many
  shops order today.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import policy
from .uncertainty import Dispersion


@dataclass
class Replay:
    on_hand: np.ndarray        # (n, D) units on the shelf at the end of each day
    sold: np.ndarray           # (n, D)
    lost: np.ndarray           # (n, D) demand that found an empty shelf
    ordered: np.ndarray        # (n, D) units ordered at the start of each day
    arrived: np.ndarray        # (n, D) units received at the start of each day

    def kpis(self, unit_cost=None, price=None):
        n, D = self.sold.shape
        dem = self.sold + self.lost
        c = np.ones(n) if unit_cost is None else np.asarray(unit_cost, float)
        p = np.ones(n) if price is None else np.asarray(price, float)
        tot = dem.sum()
        val = float((dem.sum(axis=1) * p).sum())
        return {
            "fill_rate": float(self.sold.sum() / tot) if tot else 1.0,
            "value_fill": float((self.sold.sum(axis=1) * p).sum() / val) if val else 1.0,
            "lost_units": float(self.lost.sum()),
            "lost_sales_value": float((self.lost.sum(axis=1) * p).sum()),
            "sales_value": float((self.sold.sum(axis=1) * p).sum()),
            "avg_stock_value": float((self.on_hand.mean(axis=1) * c).sum()),
            "avg_stock_units": float(self.on_hand.mean(axis=1).sum()),
            "days_with_stockout": int((self.lost > 0).sum()),
            "orders": int((self.ordered > 0).sum()),
            "units_ordered": float(self.ordered.sum()),
        }


def run(demand, lead, review, decide, on_hand0, pipeline=None) -> Replay:
    """demand (n, D) units wanted per day; lead (n,) days; review (n, D) True where an order may be placed at the
    start of that day; decide(day, rows, on_hand, position) -> units to order for those rows."""
    demand = np.asarray(demand, float)
    n, D = demand.shape
    lead = np.asarray(lead, int)
    due = np.zeros((n, D + int(lead.max()) + 1))
    if pipeline is not None:                         # (n, k) units arriving on days 0..k-1
        due[:, :pipeline.shape[1]] += pipeline
    oh = np.asarray(on_hand0, float).copy()
    out = Replay(np.zeros((n, D)), np.zeros((n, D)), np.zeros((n, D)), np.zeros((n, D)), np.zeros((n, D)))
    rows_all = np.arange(n)
    for d in range(D):
        arr = due[:, d]
        oh += arr
        out.arrived[:, d] = arr
        rows = rows_all[review[:, d]]
        if len(rows):
            position = oh[rows] + due[rows, d + 1:].sum(axis=1)
            q = np.maximum(np.asarray(decide(d, rows, oh[rows], position), float), 0)
            out.ordered[rows, d] = q
            when = d + lead[rows]
            now = when == d                                  # same-day delivery
            oh[rows[now]] += q[now]
            out.arrived[rows[now], d] += q[now]
            due[rows[~now], when[~now]] += q[~now]
        s = np.minimum(oh, demand[:, d])
        out.sold[:, d] = s
        out.lost[:, d] = demand[:, d] - s
        oh -= s
        out.on_hand[:, d] = oh
    return out


class Planner:
    """Stock Planner's way: on each order day, every unit whose chance of selling before the next delivery is worth
    the money it ties up (shadow price lam, see policy.shadow_price), in whole packs. mu: (n, D + extra) forecast mean
    per day from day 0 of the replay, made with sales up to the day before each decision (lags of at least the lead
    time + review period). w: value of a sale, c: value of a unit of stock, keep: units to keep at least."""

    def __init__(self, mu, disp: Dispersion, lead, review_days, lam, w, c, pack=1, moq=0, keep=0, recent=None,
                 level_weight=None):
        self.mu, self.disp = np.asarray(mu, float), disp
        n = self.mu.shape[0]
        # recent = (actual, forecast, h0): sales and the forecasts made for them, column h0 + d being day d; on day d
        # the 28 days before it say how far the forecast has been off lately (see level_factor)
        self.recent = recent
        self.level_weight = LEVEL_WEIGHT if level_weight is None else level_weight
        b = lambda v, t=float: np.broadcast_to(np.asarray(v, t), (n,))       # noqa: E731
        self.lead, self.rv = b(lead, int), b(review_days, int)
        self.lam, self.w, self.c = b(lam), b(w), b(c)
        self.pack, self.moq, self.keep = b(pack), b(moq), b(keep)
        self.c_mu = np.concatenate([np.zeros((n, 1)), np.cumsum(self.mu, axis=1)], axis=1)
        self.levels = []

    def demand_ahead(self, d, rows):
        L, R = self.lead[rows], self.rv[rows]
        c = self.c_mu
        end_l = np.minimum(d + L, self.mu.shape[1])
        end_lr = np.minimum(d + L + R, self.mu.shape[1])
        f = 1.0
        if self.recent is not None:
            act, fc, h0 = self.recent
            f = level_factor(act[rows, max(h0 + d - 28, 0):h0 + d], fc[rows, max(h0 + d - 28, 0):h0 + d],
                             self.level_weight)
        m_l = (c[rows, end_l] - c[rows, d]) * f
        m_lr = (c[rows, end_lr] - c[rows, d]) * f
        per_day = m_lr / np.maximum(L + R, 1)
        return (m_l, self.disp.phi(per_day, np.maximum(L, 1)), m_lr, self.disp.phi(per_day, L + R),
                self.disp.phi(per_day, R))

    def start(self, rows=None):
        """Order-up-to levels on day 0 (to start a replay as if this way had been running for a while)."""
        rows = np.arange(self.mu.shape[0]) if rows is None else rows
        m_l, phi_l, m_lr, phi_lr, _ = self.demand_ahead(0, rows)
        ss = policy.SteadyState(m_l, phi_l, m_lr, phi_lr)
        S, _ = ss.levels(np.clip(self.lam[rows] * self.c[rows] / np.maximum(self.w[rows], 1e-12), 0, 1))
        return np.maximum(S, self.keep[rows])

    def __call__(self, d, rows, on_hand, position):
        m_l, phi_l, m_lr, phi_lr, phi_r = self.demand_ahead(d, rows)
        floor = np.maximum(np.ceil(self.keep[rows] - position), 0).astype(int)
        q = policy.order_units(position, m_l, phi_l, m_lr, phi_lr, phi_r, self.w[rows], self.c[rows],
                               self.lam[rows], self.pack[rows].astype(int), self.moq[rows].astype(int), floor)
        self.levels.append((d, rows, position + q))
        return q


LEVEL_WEIGHT = 0.25     # chosen on the M5 public leaderboard period (benchmarks/run_m5.py), before the final test


def level_factor(actual, forecast, weight=None, prior=2.0):
    """Correction of the forecast with the latest weeks' sales. The forecast's inputs are at least a horizon old, so it
    has not seen the last weeks; how their actual sales compare with what was forecast for them (with `prior` units
    on each side, so a few sales do not swing it) moves the forecast part of the way (`weight`), between half and
    double."""
    weight = LEVEL_WEIGHT if weight is None else weight
    a = np.asarray(actual, float).sum(axis=-1)
    f = np.asarray(forecast, float).sum(axis=-1)
    return np.clip((1 - weight) + weight * (a + prior) / (f + prior), 0.5, 2.0)


def sample_demand(mu, disp: Dispersion, days, reps, rng):
    """Possible futures for every product, drawn around the forecast with the measured ranges: (n * reps, days),
    reps futures per product (product i's futures are rows i, i + n, i + 2n, ...). Each future has its own level (the
    forecast can be off for the whole period) on top of day-to-day noise, as in uncertainty.sample_paths."""
    mu = np.asarray(mu, float)[:, :days]
    M, Q = mu.sum(axis=1), (mu ** 2).sum(axis=1)
    per = M / days
    phi_d = np.maximum(disp.phi(per, 1), 1e-6)
    phi_w = disp.phi(per, min(days, 28))
    v = np.maximum((phi_w * M * M - phi_d * Q) / np.maximum(M * M + phi_d * Q, 1e-12), 0)
    V = np.tile(v, reps)
    level = np.where(V > 1e-6, rng.gamma(1 / np.maximum(V, 1e-6), np.maximum(V, 1e-6)), 1.0)
    mm = np.tile(mu, (reps, 1)) * level[:, None]
    pd_ = np.tile(phi_d, reps)[:, None]
    return rng.poisson(rng.gamma(1 / pd_, np.maximum(mm, 1e-12) * pd_)).astype(float)


def calibrate(mu, disp: Dispersion, lead, review_days, w, c, target, days=None, reps=1, pack=1, moq=0, keep=0,
              offset=0, seed=0, iters=10):
    """Shadow price whose orders serve `target` of the demand value: Stock Planner is run on possible futures drawn
    from the forecast and its ranges (lost sales, review days, packs and minimums included), and lam is searched until
    the simulated share served meets the target. reps: futures per product (more for small shops)."""
    mu = np.asarray(mu, float)
    n = mu.shape[0]
    b = lambda v, t=float: np.broadcast_to(np.asarray(v, t), (n,))           # noqa: E731
    lead, review_days = b(lead, int), b(review_days, int)
    w, c, pack, moq, keep, offset = b(w), b(c), b(pack), b(moq), b(keep), b(offset, int)
    days = int(days or max(3 * int((lead + review_days).max()), 28))
    days = min(days, mu.shape[1] - int((lead + review_days).max()))
    rng = np.random.default_rng(seed)
    dem = sample_demand(mu, disp, days, reps, rng)
    t = lambda v: np.tile(v, reps)                                            # noqa: E731
    mu_r = np.tile(mu, (reps, 1))
    rv = review_mask(n * reps, days, t(review_days), t(offset))
    wt = t(w)
    tot = float((dem.sum(axis=1) * wt).sum())
    if tot <= 0:
        return 0.0

    def fill(lam):
        pl = Planner(mu_r, disp, t(lead), t(review_days), lam, wt, t(c), t(pack), t(moq), t(keep))
        r = run(dem, t(lead), rv, pl, pl.start())
        return float((r.sold.sum(axis=1) * wt).sum()) / tot

    hi = float(np.max(w / np.maximum(c, 1e-12)))
    lo = hi * 1e-5
    if fill(lo) < target:
        return lo
    if fill(hi) >= target:
        return hi
    for _ in range(iters):                          # bisection on log(lam): service falls as lam rises
        mid = float(np.sqrt(lo * hi))
        if fill(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def calibrate_steady(mu, disp: Dispersion, lead, review_days, w, c, target):
    """Shadow price that serves `target` of the demand value in steady state (backorder approximation, quick)."""
    pl = Planner(mu, disp, lead, review_days, 0.0, w, c)
    n = pl.mu.shape[0]
    m_l, phi_l, m_lr, phi_lr, _ = pl.demand_ahead(0, np.arange(n))
    lam, _ = policy.shadow_price(policy.SteadyState(m_l, phi_l, m_lr, phi_lr), pl.w, pl.c, target)
    return lam


class CoverRule:
    """Rule of thumb: when stock (with what is on order) covers fewer than `low` days of average sales of the last 28
    days, order up to `high` days of cover. history: (n, H) sales before day 0; sales so far are added as the replay
    runs (pass the replay's `demand` as `seen` for days already past, which is what the shop knows)."""

    def __init__(self, history, seen, low, high, pack=1, moq=0, window=28):
        self.hist = np.concatenate([np.asarray(history, float), np.asarray(seen, float)], axis=1)
        self.h0 = np.asarray(history).shape[1]
        n = self.hist.shape[0]
        self.low = np.broadcast_to(np.asarray(low, float), (n,))
        self.high = np.broadcast_to(np.asarray(high, float), (n,))
        self.pack = np.broadcast_to(np.asarray(pack, float), (n,))
        self.moq = np.broadcast_to(np.asarray(moq, float), (n,))
        self.window = window

    def avg(self, d, rows):
        e = self.h0 + d                                     # days before the decision only
        return self.hist[rows, max(e - self.window, 0):e].mean(axis=1)

    def __call__(self, d, rows, on_hand, position):
        a = self.avg(d, rows)
        q = np.where(position < self.low[rows] * a, np.ceil(self.high[rows] * a - position), 0)
        return policy.round_order(q, self.pack[rows], self.moq[rows])


def review_mask(n, D, review_days, offset=0):
    """True on the days an order is placed: every review_days days from day `offset` (per product)."""
    rv = np.broadcast_to(np.asarray(review_days, int), (n,))
    off = np.broadcast_to(np.asarray(offset, int), (n,))
    d = np.arange(D)[None, :]
    return ((d - off[:, None]) % rv[:, None] == 0) & (d >= off[:, None])
