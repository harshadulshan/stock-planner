"""
How much to order: periodic review, lost sales, one service target for the whole shop.

Every R days an order can be placed with a supplier; it arrives L days later. A customer who finds an empty shelf
leaves (a lost sale). With D_x the demand over the next x days (negative binomial around the forecast, see
uncertainty.py) the stock when today's order lands is (avail - D_L)+, and the k-th unit ordered now sells before the
next delivery if the demand of the review period reaches (avail - D_L)+ + k.

One target for the whole shop: serve a share of the demand (in sales value) from the shelf with as little money in
stock as possible. The answer is a single price on money tied up in stock, lam: a unit is worth ordering while its
price times its chance of selling before the next delivery is at least lam times its cost. Fast, steady sellers get
deep stock, slow and erratic ones only what is likely to sell; lam is set so that the whole shop reaches the target
(Lagrange relaxation of "least stock for a given expected fill rate", solved product by product).

Orders are in whole packs and at least the minimum order quantity. With a budget, the packs that sell the most value
per unit of money go first; an order below a supplier's minimum order value is topped up when something is urgent,
otherwise it waits for the next order day.

Also here: the classic per-product fill-rate levels (order_up_to, order_qty), used for checks and comparisons.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from .uncertainty import nb

EXACT_MAX = 400.0          # mean demand above this: normal approximation


def _pmf_table(mean, phi, K):
    """P(D = j) for j = 0..K-1 of NB(mean, phi) per row: (n, K), from log-gamma (fast, no underflow)."""
    mean = np.maximum(np.asarray(mean, float), 1e-9)[:, None]
    r = 1.0 / np.maximum(np.asarray(phi, float), 1e-6)[:, None]
    j = np.arange(K)[None, :]
    lp = (gammaln(j + r) - gammaln(r) - gammaln(j + 1) + r * np.log(r / (r + mean))
          + j * np.log(mean / (r + mean)))
    return np.exp(lp)


def _cdf_table(mean, phi, K):
    """CDF F(j) for j = 0..K-1 of NB(mean, phi) per row: (n, K)."""
    return np.minimum(np.cumsum(_pmf_table(mean, phi, K), axis=1), 1.0)


def _C(F):
    """C(S) = sum_{j<S} F(j) for S = 0..K: (n, K+1)."""
    return np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum(F, axis=1)], axis=1)


def shortage_table(mean, phi, K):
    """E[(D - S)+] for S = 0..K (exact negative binomial)."""
    mean = np.asarray(mean, float)
    C = _C(_cdf_table(mean, np.asarray(phi, float), K))
    S = np.arange(K + 1)[None, :]
    return np.maximum(mean[:, None] - S + C, 0.0)


def _normal_loss(mean, sd, S):
    z = (S - mean) / np.maximum(sd, 1e-9)
    return np.maximum(sd * stats.norm.pdf(z) - (S - mean) * stats.norm.sf(z), 0.0)


def cycle_short(S, m_lr, phi_lr, m_l, phi_l):
    """Expected units short per cycle with stock position S (any shape, broadcast)."""
    S = np.asarray(S, float)
    out = np.zeros(np.broadcast(S, m_lr).shape)
    m_lr, phi_lr, m_l, phi_l, S = np.broadcast_arrays(m_lr, phi_lr, m_l, phi_l, S)
    for k in np.ndindex(out.shape):
        out[k] = _short_one(S[k], m_lr[k], phi_lr[k], m_l[k], phi_l[k])
    return out


def _short_one(S, m_lr, phi_lr, m_l, phi_l):
    if m_lr <= 1e-12:
        return 0.0
    if m_lr > EXACT_MAX:
        a = _normal_loss(m_lr, np.sqrt(m_lr + phi_lr * m_lr ** 2), S)
        b = _normal_loss(m_l, np.sqrt(m_l + phi_l * m_l ** 2), S) if m_l > 0 else 0.0
        return float(max(a - b, 0.0))
    K = int(S) + 1
    a = shortage_table(np.array([m_lr]), np.array([phi_lr]), K)[0, int(S)]
    b = shortage_table(np.array([m_l]), np.array([phi_l]), K)[0, int(S)] if m_l > 0 else max(0.0, -S)
    return float(max(a - b, 0.0))


def order_up_to(m_lr, phi_lr, m_l, phi_l, target, batch=2000):
    """Smallest stock position S (whole units) with fill rate >= target in a cycle, per product."""
    m_lr = np.asarray(m_lr, float)
    m_l = np.asarray(m_l, float)
    phi_lr = np.broadcast_to(np.asarray(phi_lr, float), m_lr.shape)
    phi_l = np.broadcast_to(np.asarray(phi_l, float), m_lr.shape)
    target = np.broadcast_to(np.asarray(target, float), m_lr.shape)
    m_r = np.maximum(m_lr - m_l, 0)
    out = np.zeros(m_lr.shape, int)
    need = m_r > 1e-9
    exact = need & (m_lr <= EXACT_MAX)
    big = need & (m_lr > EXACT_MAX)
    idx = np.flatnonzero(exact)
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        K = int(np.max(nb(m_lr[ii], phi_lr[ii]).ppf(0.99999))) + 2
        C_lr = _C(_cdf_table(m_lr[ii], phi_lr[ii], K))
        C_l = _C(_cdf_table(np.maximum(m_l[ii], 1e-9), phi_l[ii], K))
        S = np.arange(K + 1)[None, :]
        short = np.maximum(m_lr[ii, None] - S + C_lr, 0) - np.where(m_l[ii, None] > 1e-9,
                                                                     np.maximum(m_l[ii, None] - S + C_l, 0), 0)
        fill = 1 - np.maximum(short, 0) / m_r[ii, None]
        ok = fill >= target[ii, None] - 1e-12
        out[ii] = np.where(ok.any(axis=1), ok.argmax(axis=1), K)
    for i in np.flatnonzero(big):                   # normal approximation, bisection
        sd_lr = np.sqrt(m_lr[i] + phi_lr[i] * m_lr[i] ** 2)
        sd_l = np.sqrt(m_l[i] + phi_l[i] * m_l[i] ** 2)
        lo, hi = 0.0, m_lr[i] + 12 * sd_lr
        for _ in range(60):
            mid = (lo + hi) / 2
            sh = _normal_loss(m_lr[i], sd_lr, mid) - (_normal_loss(m_l[i], sd_l, mid) if m_l[i] > 0 else 0)
            if 1 - sh / m_r[i] >= target[i]:
                hi = mid
            else:
                lo = mid
        out[i] = int(np.ceil(hi))
    return out


def order_qty(avail, m_l, phi_l, m_lr, phi_lr, phi_r, target, batch=1000):
    """Units to order now so that, after the order arrives, the shelf serves at least `target` of the demand until
    the next order arrives, for a shop where a customer who finds an empty shelf leaves (lost sales).

    avail: stock on hand plus what is on order (arriving before this order). Demand in the lead time D_L uses up
    the stock; what it cannot get is lost, so the stock when the order lands is (avail - D_L)+ + q, and the units
    short until the next delivery are E[(D_R - (avail - D_L)+ - q)+], with D_R the demand of the review period
    (negative binomial, independent of D_L). Where running out before the order lands is practically impossible
    this equals the order-up-to formula, which is used there (it also keeps the measured link between D_L and D_R).
    """
    avail = np.maximum(np.asarray(avail, float), 0)
    m_l = np.asarray(m_l, float)
    m_lr = np.asarray(m_lr, float)
    shape = m_lr.shape
    phi_l, phi_lr, phi_r, target = (np.broadcast_to(np.asarray(x, float), shape)
                                    for x in (phi_l, phi_lr, phi_r, target))
    m_r = np.maximum(m_lr - m_l, 0)
    S = order_up_to(m_lr, phi_lr, m_l, phi_l, target)
    q = np.maximum(S - avail, 0).astype(float)
    a_int = np.floor(avail + 1e-9).astype(int)
    risky = (m_r > 1e-9) & (m_l > 1e-9)
    if risky.any():
        p_run_out = np.zeros(shape)
        p_run_out[risky] = nb(m_l[risky], phi_l[risky]).sf(a_int[risky] - 1)     # P(D_L >= avail)
        risky &= p_run_out > 1e-6
    idx = np.flatnonzero(risky)
    for s0 in range(0, len(idx), batch):
        ii = idx[s0:s0 + batch]
        A = a_int[ii]
        KA = int(A.max()) + 1
        x = np.arange(KA)[None, :]
        dl = nb(m_l[ii, None], phi_l[ii, None])
        pX = np.where(x == 0, dl.sf(A[:, None] - 1), np.where(x <= A[:, None], dl.pmf(A[:, None] - x), 0.0))
        sd_r = np.sqrt(m_r[ii] + phi_r[ii] * m_r[ii] ** 2)
        hi = np.ceil(q[ii] + 3 * sd_r + 3).astype(int)
        K = KA + int(hi.max()) + 1
        lossR = shortage_table(m_r[ii], phi_r[ii], K)                # E[(D_R - s)+], s = 0..K
        rows = np.arange(len(ii))[:, None]

        def fill_at(qq):
            short = (pX * lossR[rows, x + qq[:, None]]).sum(axis=1)
            return 1 - short / m_r[ii]

        lo = np.zeros(len(ii), int)
        hi_ = hi.copy()
        ok_hi = fill_at(hi_) >= target[ii] - 1e-12
        hi_ = np.where(ok_hi, hi_, hi_)                          # if even hi misses, keep hi (bounded effort)
        for _ in range(int(np.ceil(np.log2(max(int(hi.max()), 1) + 1))) + 1):
            mid = (lo + hi_) // 2
            good = fill_at(mid) >= target[ii] - 1e-12
            hi_ = np.where(good, mid, hi_)
            lo = np.where(good, lo, mid + 1)
        lo = np.minimum(lo, hi_)
        q[ii] = np.where(fill_at(lo) >= target[ii] - 1e-12, lo, hi_)
    q[m_r <= 1e-9] = 0
    return q


def cycle_fill(avail, q, m_l, phi_l, m_lr, phi_lr, phi_r):
    """Expected share of the review period's demand served when `q` is ordered now (lost sales, as order_qty)."""
    avail = np.atleast_1d(np.asarray(avail, float))
    q = np.broadcast_to(np.asarray(q, float), avail.shape)
    out = np.ones(avail.shape)
    for k in range(avail.size):
        ml, mlr = float(np.atleast_1d(m_l)[k]), float(np.atleast_1d(m_lr)[k])
        mr = max(mlr - ml, 0.0)
        if mr <= 1e-9:
            continue
        pl, plr, pr = (float(np.atleast_1d(v)[k]) for v in (phi_l, phi_lr, phi_r))
        A = int(np.floor(avail[k] + 1e-9))
        if ml <= 1e-9 or nb(ml, pl).sf(A - 1) <= 1e-6:
            out[k] = 1 - _short_one(A + q[k], mlr, plr, ml, pl) / mr
            continue
        dl = nb(ml, pl)
        x = np.arange(A + 1)
        pX = np.where(x == 0, dl.sf(A - 1), dl.pmf(A - x))
        lossR = shortage_table(np.array([mr]), np.array([pr]), A + int(q[k]) + 1)[0]
        out[k] = 1 - float((pX * lossR[x + int(q[k])]).sum()) / mr
    return out


def fill_rate(S, m_lr, phi_lr, m_l, phi_l):
    m_r = np.maximum(np.asarray(m_lr, float) - np.asarray(m_l, float), 0)
    sh = cycle_short(S, m_lr, phi_lr, m_l, phi_l)
    return np.where(m_r > 1e-12, 1 - sh / np.maximum(m_r, 1e-12), 1.0)


def round_order(q, pack, moq):
    """Order quantity rounded up to whole packs and to the minimum order quantity (0 stays 0)."""
    q = np.asarray(q, float)
    pack = np.maximum(np.asarray(pack, float), 1)
    moq = np.maximum(np.asarray(moq, float), 0)
    r = np.ceil(np.maximum(q, 0) / pack - 1e-9) * pack
    r = np.where(r > 0, np.maximum(r, np.ceil(moq / pack - 1e-9) * pack), 0)
    return r.astype(int)


# ------------------------------------------------------------------------------------------------ shop-wide service
class SteadyState:
    """For every product and every order-up-to level S: the expected sales of one review period (periodic review,
    backorder approximation). Used to find the price of tying money up in stock that meets a shop-wide target."""

    def __init__(self, m_l, phi_l, m_lr, phi_lr, cells=2_000_000):
        m_l, m_lr = np.asarray(m_l, float), np.asarray(m_lr, float)
        phi_l = np.broadcast_to(np.asarray(phi_l, float), m_lr.shape)
        phi_lr = np.broadcast_to(np.asarray(phi_lr, float), m_lr.shape)
        self.n = len(m_lr)
        self.m_r = np.maximum(m_lr - m_l, 0)
        K = np.where(m_lr > 1e-9, nb(np.maximum(m_lr, 1e-9), phi_lr).ppf(0.99999), 0).astype(int) + 2
        order = np.argsort(K)
        self.parts = []
        s = 0
        while s < self.n:
            e = s + 1
            while e < self.n and (e + 1 - s) * K[order[e]] <= cells:
                e += 1
            ii = order[s:e]
            k = int(K[ii].max())
            C_lr = _C(_cdf_table(np.maximum(m_lr[ii], 1e-9), phi_lr[ii], k))
            C_l = _C(_cdf_table(np.maximum(m_l[ii], 1e-9), phi_l[ii], k))
            S = np.arange(k + 1)[None, :]
            G = np.maximum(m_lr[ii, None] - S + C_lr, 0) - np.where(m_l[ii, None] > 1e-9,
                                                                     np.maximum(m_l[ii, None] - S + C_l, 0), 0)
            ES = np.maximum(self.m_r[ii, None] - G, 0)            # expected sales of the review period
            self.parts.append((ii, ES))
            s = e

    def levels(self, theta):
        """Order-up-to level maximising expected sales - theta * S per product (theta = least chance of selling that
        is worth a unit of stock), and the expected sales there."""
        theta = np.broadcast_to(np.asarray(theta, float), (self.n,))
        S = np.zeros(self.n, int)
        es = np.zeros(self.n)
        for ii, ES in self.parts:
            score = ES - theta[ii, None] * np.arange(ES.shape[1])[None, :]
            k = np.argmax(score > score.max(axis=1, keepdims=True) - 1e-12, axis=1)
            S[ii] = k
            es[ii] = ES[np.arange(len(ii)), k]
        return S, es


def shadow_price(ss: SteadyState, w, c, target, iters=50):
    """lam such that stocking every unit whose chance of selling in the review period is at least lam * c / w
    serves `target` of the demand value (w = value of a sale, c = value of a unit of stock). Returns (lam, theta)."""
    w = np.broadcast_to(np.asarray(w, float), (ss.n,))
    c = np.broadcast_to(np.asarray(c, float), (ss.n,))
    tot = float((w * ss.m_r).sum())
    if tot <= 0:
        return 0.0, np.zeros(ss.n)

    def fill(lam):
        _, es = ss.levels(np.clip(lam * c / np.maximum(w, 1e-12), 0, 1))
        return float((w * es).sum()) / tot

    lo, hi = 0.0, float(np.max(w / np.maximum(c, 1e-12)))
    if fill(lo) < target:
        return 0.0, np.zeros(ss.n)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if fill(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, np.clip(lo * c / np.maximum(w, 1e-12), 0, 1)


def _prep(avail, m_l, phi_l, m_lr, phi_lr, phi_r):
    avail = np.floor(np.maximum(np.asarray(avail, float), 0) + 1e-9).astype(int)
    m_l, m_lr = np.asarray(m_l, float), np.asarray(m_lr, float)
    shape = m_lr.shape
    phi_l, phi_lr, phi_r = (np.broadcast_to(np.asarray(x, float), shape) for x in (phi_l, phi_lr, phi_r))
    return avail, m_l, phi_l, m_lr, phi_lr, phi_r


def _probs(ii, avail, m_l, phi_l, m_lr, phi_lr, phi_r, K):
    """Chances of selling for units 1..K of products ii (see sell_probs)."""
    m_r = np.maximum(m_lr - m_l, 0)
    P = np.zeros((len(ii), K))
    if K == 0:
        return P
    k = np.arange(1, K + 1)[None, :]
    has = m_r[ii] > 1e-9
    risky = has & (m_l[ii] > 1e-9)
    if risky.any():
        r_ = np.zeros(len(ii), bool)
        jj = ii[risky]
        r_[risky] = nb(m_l[jj], phi_l[jj]).sf(avail[jj] - 1) > 1e-6
        risky = r_
    pl = np.flatnonzero(has & ~risky)
    if len(pl):                                             # P(D_L < S <= D_LR) for S = avail + k
        g = ii[pl]
        top = int(avail[g].max()) + K
        F_lr = _cdf_table(m_lr[g], phi_lr[g], top)
        F_l = _cdf_table(np.maximum(m_l[g], 1e-9), phi_l[g], top)
        cols = avail[g][:, None] + k - 1                    # P(D >= s + 1) = 1 - F(s), s = avail + k - 1
        rows = np.arange(len(g))[:, None]
        sf_lr = 1 - F_lr[rows, cols]
        sf_l = np.where(m_l[g, None] > 1e-9, 1 - F_l[rows, cols], 0)
        P[pl] = np.clip(sf_lr - sf_l, 0, 1)
    rk = np.flatnonzero(risky)
    if len(rk):
        g = ii[rk]
        A = avail[g]
        KA = int(A.max()) + 1
        x = np.arange(KA)[None, :]
        pmf_l = _pmf_table(m_l[g], phi_l[g], KA + 1)
        cdf_l = np.minimum(np.cumsum(pmf_l, axis=1), 1.0)
        rows = np.arange(len(g))[:, None]
        sf_at = 1 - np.where(A[:, None] >= 1, cdf_l[rows, np.maximum(A[:, None] - 1, 0)], 0)   # P(D_L >= A)
        pX = np.where(x == 0, sf_at, np.where(x <= A[:, None], pmf_l[rows, np.clip(A[:, None] - x, 0, KA)], 0.0))
        sfR = 1 - _cdf_table(m_r[g], phi_r[g], KA + K)      # sfR[s] = P(D_R >= s + 1)
        win = np.lib.stride_tricks.sliding_window_view(sfR, KA, axis=1)[:, :K, :]
        P[rk] = np.einsum("nkx,nx->nk", win, pX)           # P(D_R >= x + k) = sfR[x + k - 1]
    return P


def _need(avail, m_lr, phi_lr):
    hi = np.where(m_lr > 1e-9, nb(np.maximum(m_lr, 1e-9), phi_lr).ppf(1 - 1e-6), 0)
    return np.maximum(hi - avail + 2, 1).astype(int)


def sell_probs(avail, m_l, phi_l, m_lr, phi_lr, phi_r):
    """P[i, k-1]: chance that the k-th unit ordered now is sold before the next delivery (k = 1..K), for a shop where
    an empty shelf loses the sale: the stock when the order lands is (avail - D_L)+, so the k-th unit sells if the
    demand of the review period reaches (avail - D_L)+ + k. Where running out before the order lands is practically
    impossible this is P(D_L < avail + k <= D_LR), keeping the measured link between the two. For a few products."""
    args = _prep(avail, m_l, phi_l, m_lr, phi_lr, phi_r)
    K = int(_need(args[0], args[3], args[4]).max()) if len(args[3]) else 1
    return _probs(np.arange(len(args[3])), *args, K)


def order_units(avail, m_l, phi_l, m_lr, phi_lr, phi_r, w, c, lam, pack=1, moq=0, floor=0, cells=2_000_000):
    """Units to order for many products at once (see decide), working through them in batches of similar size."""
    args = _prep(avail, m_l, phi_l, m_lr, phi_lr, phi_r)
    n = len(args[3])
    need = _need(args[0], args[3], args[4])
    bc = lambda v, t=float: np.broadcast_to(np.asarray(v, t), (n,))        # noqa: E731
    w, c, lam, pack, moq, floor = bc(w), bc(c), bc(lam), bc(pack, int), bc(moq, int), bc(floor, int)
    out = np.zeros(n, int)
    order = np.argsort(need)
    s = 0
    while s < n:
        e = s + 1
        while e < n and (e + 1 - s) * need[order[e]] <= cells:
            e += 1
        ii = order[s:e]
        P = _probs(ii, *args, int(need[ii].max()))
        out[ii] = decide(P, w[ii], c[ii], lam[ii], pack[ii], moq[ii], floor[ii])
        s = e
    return out


def decide(P, w, c, lam, pack=1, moq=0, floor=0):
    """Units to order: whole packs, at least the minimum order quantity, maximising the expected sales value of the
    units (w * chance of selling) minus lam * c for each unit; floor = units needed to keep at least that many."""
    n, K = P.shape
    if n == 0:
        return np.zeros(0, int)
    w = np.broadcast_to(np.asarray(w, float), (n,))
    c = np.broadcast_to(np.asarray(c, float), (n,))
    lam = np.broadcast_to(np.asarray(lam, float), (n,))
    pack = np.maximum(np.broadcast_to(np.asarray(pack, int), (n,)), 1)
    moq = np.broadcast_to(np.asarray(moq, int), (n,))
    floor = np.broadcast_to(np.asarray(floor, int), (n,))
    gain = w[:, None] * P - (lam * c)[:, None]
    G = np.concatenate([np.zeros((n, 1)), np.cumsum(gain, axis=1)], axis=1)          # value of ordering Q units
    first = np.ceil(np.maximum(moq, 1) / pack).astype(int) * pack
    if (pack == 1).all() and (moq <= 1).all() and (floor <= 0).all():                 # single units: best Q directly
        return np.argmax(G > G.max(axis=1, keepdims=True) - 1e-9, axis=1)
    out = np.zeros(n, int)
    for i in range(n):
        top = max(K, int(floor[i]))
        cands = [0] + list(range(int(first[i]), top + int(pack[i]) + 1, int(pack[i])))
        cands = [q for q in cands if q >= floor[i]] or [int(first[i]) + int(pack[i]) * int(
            np.ceil(max(floor[i] - first[i], 0) / pack[i]))]
        val = [G[i, min(q, K)] - (max(q - K, 0) * lam[i] * c[i]) for q in cands]
        out[i] = cands[int(np.argmax(np.array(val) > max(val) - 1e-9))]
    return out


# ------------------------------------------------------------------------------------------------ order today
def plan_orders(items: pd.DataFrame, lam, budget=None, supplier_min=None) -> pd.DataFrame:
    """What to order today.

    items: one row per product that can be ordered today, with columns
        position (on hand + on order), on_hand, m_l, m_lr, phi_l, phi_lr, phi_r (demand over the lead time, over lead
        time plus review period, and dispersion of the review period alone), pack, moq, unit_cost, price, supplier,
        keep (units to keep on the shelf at least, e.g. 1 for every product sold lately)
    lam: shadow price of stock (policy.shadow_price): a unit is worth ordering while price * its chance of selling
        before the next delivery is at least lam * unit cost
    budget: money available for all orders today (None = no limit)
    supplier_min: {supplier: minimum order value}
    Returns the items with: qty, value, last_unit_chance, fill_without, fill_with, risk_before_arrival, note.
    """
    it = items.copy().reset_index(drop=True)
    n = len(it)
    pos = it["position"].to_numpy(float)
    w = it["price"].to_numpy(float)
    cost = it["unit_cost"].to_numpy(float)
    keep = it["keep"].to_numpy(float) if "keep" in it else np.zeros(n)
    P = sell_probs(pos, it["m_l"], it["phi_l"], it["m_lr"], it["phi_lr"], it["phi_r"])
    K = P.shape[1]
    floor = np.maximum(np.ceil(keep - pos), 0).astype(int)
    want = decide(P, w, cost, lam, it["pack"].to_numpy(int), it["moq"].to_numpy(int), floor)
    cumP = np.concatenate([np.zeros((n, 1)), np.cumsum(P, axis=1)], axis=1)
    note = [""] * n

    def es(i, q):
        """Expected units sold until the next delivery out of q ordered now."""
        return float(cumP[i, min(int(q), K)])

    def steps(i, upto):
        """Order sizes for product i: the smallest allowed (whole packs, minimum, floor), then one pack at a time."""
        p = max(int(it.at[i, "pack"]), 1)
        first = int(round_order(max(1, floor[i]), p, it.at[i, "moq"]))
        out = []
        q = first
        while q <= max(upto, 0):
            out.append(q)
            q += p
        return out

    sup_of = it["supplier"].astype(str).to_numpy()
    supplier_min = {str(k): float(v) for k, v in (supplier_min or {}).items()}
    closed = set()

    def allocate():
        """Within the budget: the packs that sell the most value per unit of money first (never beyond the plan)."""
        qty = np.where(np.isin(sup_of, list(closed)), 0, want).astype(int)
        if budget is None or float((qty * cost).sum()) <= budget + 1e-9:
            return qty, (None if budget is None else float(budget) - float((qty * cost).sum()))
        qty = np.zeros(n, int)
        heap = []
        plan = {i: steps(i, int(want[i])) if sup_of[i] not in closed else [] for i in range(n)}

        def push(i, k):
            if k < len(plan[i]):
                q0 = plan[i][k - 1] if k else 0
                q1 = plan[i][k]
                gain = w[i] * (es(i, q1) - es(i, q0))
                c = (q1 - q0) * cost[i]
                if gain > 1e-9:
                    heapq.heappush(heap, (-gain / max(c, 1e-9), i, k, c))

        for i in range(n):
            push(i, 0)
        left = float(budget)
        while heap:
            _, i, k, c = heapq.heappop(heap)
            if c <= left + 1e-9:
                qty[i] = plan[i][k]
                left -= c
                push(i, k + 1)
        return qty, left

    run_out = np.where(it["m_lr"].to_numpy(float) > 1e-9,
                       nb(np.maximum(it["m_lr"].to_numpy(float), 1e-9), it["phi_lr"].to_numpy(float)).sf(pos), 0)
    why_closed = {}
    for _ in range(len(supplier_min) + 1):
        qty, left = allocate()
        note[:] = [""] * n
        for i in range(n):
            if sup_of[i] in why_closed and want[i] > 0:
                note[i] = f"waits: below {sup_of[i]}'s minimum order and {why_closed[sup_of[i]]}"
        changed = False
        for sup, mn in supplier_min.items():
            sel = np.flatnonzero(sup_of == sup)
            if sup in closed or not len(sel):
                continue
            val = float((qty[sel] * cost[sel]).sum())
            if val <= 0 or val >= mn - 1e-9:
                continue
            urgent = bool((run_out[sel] > 0.5).any())      # likely to run out before the next delivery
            reached = False
            if urgent:                                     # top up with the packs that sell the most value
                tq, tv, tl, added = qty.copy(), val, left, []
                while tv < mn - 1e-9:
                    best = None
                    for i in sel:
                        p = max(int(it.at[i, "pack"]), 1)
                        q0 = int(tq[i])
                        q1 = q0 + p if q0 > 0 else int(round_order(1, p, it.at[i, "moq"]))
                        c = (q1 - q0) * cost[i]
                        if tl is not None and c > tl + 1e-9:
                            continue                       # does not fit in what is left of the budget
                        r = w[i] * (es(i, q1) - es(i, q0)) / max(c, 1e-9)
                        if best is None or r > best[0]:
                            best = (r, i, q1, c)
                    if best is None:
                        break
                    _, i, q1, c = best
                    tq[i] = q1
                    tv += c
                    tl = None if tl is None else tl - c
                    added.append(i)
                if tv >= mn - 1e-9:
                    reached = True
                    qty[:] = tq
                    left = tl
                    for i in added:
                        note[i] = f"topped up to reach {sup}'s minimum order"
            if not reached:
                closed.add(sup)
                changed = True
                why_closed[sup] = "the budget does not reach it" if urgent else "nothing is urgent"
        if not changed:
            break
    it["qty"] = qty
    if budget is not None:
        for i in np.flatnonzero((qty < want) & (np.array(note) == "")):
            note[i] = "less than planned: budget"
    for i in np.flatnonzero((qty > 0) & (floor > 0) & (np.array(note) == "")):
        if es(i, qty[i]) < 0.5:
            note[i] = "kept on the shelf: sold in the last 3 months"
    it["value"] = it["qty"] * it["unit_cost"]
    it["last_unit_chance"] = [float(P[i, min(int(q), K) - 1]) if q > 0 and K else 0.0 for i, q in enumerate(qty)]
    args = (it["m_l"].to_numpy(float), it["phi_l"].to_numpy(float), it["m_lr"].to_numpy(float),
            it["phi_lr"].to_numpy(float), it["phi_r"].to_numpy(float))
    it["fill_without"] = cycle_fill(pos, 0.0, *args)
    it["fill_with"] = cycle_fill(pos, it["qty"].to_numpy(float), *args)
    it["risk_before_arrival"] = 1 - nb(np.maximum(it["m_l"].to_numpy(float), 1e-9),
                                       it["phi_l"].to_numpy(float)).cdf(it["on_hand"].to_numpy(float))
    it.loc[it["m_l"] <= 1e-9, "risk_before_arrival"] = 0.0
    it["note"] = note
    return it
