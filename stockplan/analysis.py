"""
Everything the app shows for one shop: forecasts with ranges, accuracy on recent weeks it had not seen, product
classes, today's orders, possible futures of one product, the replay of the last weeks, and the stock / service
trade-off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine, policy, simulate
from .shop import Shop
from .uncertainty import Dispersion, nb, sample_paths


@dataclass
class Analysis:
    shop: Shop
    Y: np.ndarray                 # (n, T) units sold per day
    start: pd.Timestamp
    H: int                        # forecast horizon (days)
    hist: engine.History
    lead: np.ndarray
    review: np.ndarray
    offset: np.ndarray            # days from today to each product's next order day
    holdout_mean: np.ndarray      # (n, H) forecasts for the last H days, made without them
    disp: Dispersion
    mu: np.ndarray                # (n, H) forecast for today .. today + H - 1
    accuracy: pd.DataFrame
    classes: pd.DataFrame
    seconds: float
    level: np.ndarray = None      # correction with the latest weeks' sales, already in mu (simulate.level_factor)

    @property
    def T(self):
        return self.Y.shape[1]

    @property
    def n(self):
        return self.Y.shape[0]


def horizon(shop: Shop) -> int:
    s = shop.suppliers
    need = int((pd.to_numeric(s["Lead time (days)"]) + pd.to_numeric(s["Order every (days)"])).max())
    return int(min(max(14, -(-need // 7) * 7), 91))


def history(shop: Shop, Y, start, H) -> engine.History:
    T = Y.shape[1]
    dates = start + pd.to_timedelta(np.arange(T + H), unit="D")
    ev = pd.Series("", index=dates)
    if len(shop.events):
        e = shop.events.copy()
        e["Date"] = pd.to_datetime(e["Date"]).dt.normalize()
        e = e[e["Date"].isin(dates)].groupby("Date")["Event"].first()
        ev.loc[e.index] = e.astype(str).to_numpy()
    events = pd.DataFrame({"event": ev.to_numpy(), "event_type": np.where(ev.to_numpy() != "", "event", "")})
    static = shop.products[["Category", "Supplier"]].astype(str).reset_index(drop=True)
    return engine.History(Y, start, static, None, events, None)


def _rounds(rows):
    return 250 if rows < 300_000 else 150


def analyse(shop: Shop, progress=None) -> Analysis:
    t0 = time.time()
    say = progress or (lambda *_: None)
    Y, start = shop.matrix()
    n, T = Y.shape
    H = horizon(shop)
    hist = history(shop, Y, start, H)
    rows = n * min(T, 730)
    first = max(H + 7, T - 1 - H - 730)
    say(0.1, "Learning from the past (checking on the last weeks)")
    hold = engine.fit(hist, H, first_day=first, last_day=T - 1 - H, rounds=_rounds(rows))
    hdays = np.arange(T - H, T)
    hm = engine.predict(hold, hist, hdays)
    actual = Y[:, T - H:]
    disp = Dispersion.fit(hm, actual)
    say(0.5, "Learning from all the sales")
    full = engine.fit(hist, H, first_day=max(H + 7, T - 730), rounds=_rounds(rows))
    mu = engine.predict(full, hist, np.arange(T, T + H))
    # accuracy on the held-out weeks: the engine and the usual "average of the last 4 weeks"
    base = np.repeat(Y[:, T - H - 28:T - H].mean(axis=1, keepdims=True), H, axis=1)
    wk = H // 7
    def weekly(x):
        return x[:, :wk * 7].reshape(n, wk, 7).sum(axis=2)
    acc = pd.DataFrame({
        "SKU": shop.products["SKU"].to_numpy(),
        "sold": actual.sum(axis=1),
        "forecast": hm.sum(axis=1),
        "error_engine": np.abs(weekly(actual) - weekly(hm)).sum(axis=1),
        "error_average": np.abs(weekly(actual) - weekly(base)).sum(axis=1),
    })
    cls = classify(shop, Y)
    lead = pd.to_numeric(shop.products["Supplier"].map(shop.suppliers.set_index("Supplier")["Lead time (days)"])) \
        .to_numpy(int)
    review = pd.to_numeric(shop.products["Supplier"].map(shop.suppliers.set_index("Supplier")["Order every (days)"])) \
        .to_numpy(int)
    nxt = pd.to_datetime(shop.products["Supplier"].map(shop.suppliers.set_index("Supplier")["Next order date"]))
    offset = np.maximum((nxt - shop.today).dt.days.to_numpy(), 0)
    w = min(28, H)
    level = simulate.level_factor(actual[:, -w:], hm[:, -w:])
    mu = mu * level[:, None]          # the model's inputs are at least H days old: bring in the latest weeks' sales
    say(1.0, "Done")
    return Analysis(shop, Y, start, H, hist, lead, review, offset, hm, disp, mu, acc, cls, time.time() - t0, level)


def classify(shop: Shop, Y) -> pd.DataFrame:
    """ABC by sales value of the last 90 days (A: the products making 80% of it, B: the next 15%, C: the rest);
    XYZ by how much weekly sales vary (X steady, Y variable, Z erratic); days of cover of the stock on hand."""
    p = shop.products
    n, T = Y.shape
    val = Y[:, -90:].sum(axis=1) * p["Price"].to_numpy(float)
    order = np.argsort(-val)
    cum = np.cumsum(val[order]) / max(val.sum(), 1e-9)
    abc = np.empty(n, dtype=object)
    abc[order] = np.where(cum <= 0.8 + 1e-9, "A", np.where(cum <= 0.95 + 1e-9, "B", "C"))
    if n:
        abc[order[0]] = "A"
    wk = Y[:, T - 182:].reshape(n, 26, 7).sum(axis=2) if T >= 182 else Y.reshape(n, -1, 1).sum(axis=2)
    cv = wk.std(axis=1) / np.maximum(wk.mean(axis=1), 1e-9)
    xyz = np.where(cv < 0.5, "X", np.where(cv < 1.0, "Y", "Z"))
    avg = Y[:, -28:].mean(axis=1)
    cover = np.where(avg > 0, p["On hand"].to_numpy(float) / np.maximum(avg, 1e-9), np.inf)
    return pd.DataFrame({"SKU": p["SKU"].to_numpy(), "ABC": abc, "XYZ": xyz, "sales_value_90d": val,
                         "avg_per_day": avg, "days_of_cover": cover,
                         "stock_value": p["On hand"].to_numpy(float) * p["Unit cost"].to_numpy(float),
                         "last_sale_days": np.where(Y.sum(axis=1) > 0,
                                                    T - 1 - np.where(Y > 0, np.arange(T), -1).max(axis=1), T)})


def keep_units(a: Analysis, keep_one=True):
    """Units to keep on the shelf at least: one of every product sold in the last 90 days (if asked)."""
    if not keep_one:
        return np.zeros(a.n)
    return (a.Y[:, -90:].sum(axis=1) > 0).astype(float)


def _ahead(a: Analysis, cycles=4):
    """The forecast stretched to `cycles` order cycles of the slowest supplier (beyond the horizon: the last weeks'
    pattern repeated), for simulating the coming weeks."""
    need = cycles * int((a.lead + a.review).max()) + int((a.lead + a.review).max())
    mu = a.mu
    if mu.shape[1] < need:
        k = need - mu.shape[1]
        mu = np.concatenate([mu, np.tile(mu[:, -7:], (1, k // 7 + 1))[:, :k]], axis=1)
    return mu


def _reps(n):
    return int(min(12, max(1, np.ceil(1000 / max(n, 1)))))


def shadow(a: Analysis, target=0.95, keep_one=True):
    """Price of money tied up in stock whose orders serve `target` of the demand value of the whole shop, found by
    running the coming weeks many times over possible futures (lost sales, order days, packs and minimums)."""
    p = a.shop.products
    return simulate.calibrate(_ahead(a), a.disp, a.lead, a.review, p["Price"].to_numpy(float),
                              p["Unit cost"].to_numpy(float), target, reps=_reps(a.n),
                              pack=p["Pack size"].to_numpy(float), moq=p["Min order qty"].to_numpy(float),
                              keep=keep_units(a, keep_one), offset=a.offset)


def demand_ahead(a: Analysis, rows=None, start=0):
    """Mean and dispersion of demand over the lead time and over lead time + review period, from day `start`."""
    rows = np.arange(a.n) if rows is None else np.asarray(rows)
    L, R = a.lead[rows], a.review[rows]
    c = np.concatenate([np.zeros((a.n, 1)), np.cumsum(a.mu, axis=1)], axis=1)[rows]
    s = np.full(len(rows), start)
    e_l = np.minimum(s + L, a.H)
    e_lr = np.minimum(s + L + R, a.H)
    r_ = np.arange(len(rows))
    m_l = c[r_, e_l] - c[r_, s]
    m_lr = c[r_, e_lr] - c[r_, s]
    per = m_lr / np.maximum(L + R, 1)
    return m_l, a.disp.phi(per, np.maximum(L, 1)), m_lr, a.disp.phi(per, L + R), a.disp.phi(per, R)


def order_today(a: Analysis, target=0.95, budget=None, keep_one=True, lam=None) -> pd.DataFrame:
    """Recommended orders for the suppliers whose order day is today, with the reasons; other products get their
    next order date and the risk of running out before then."""
    p = a.shop.products.reset_index(drop=True)
    m_l, phi_l, m_lr, phi_lr, phi_r = demand_ahead(a)
    items = pd.DataFrame({
        "SKU": p["SKU"], "Product": p["Product"], "Category": p["Category"], "supplier": p["Supplier"],
        "on_hand": p["On hand"].astype(float), "on_order": p["On order"].astype(float),
        "position": p["On hand"].astype(float) + p["On order"].astype(float),
        "m_l": m_l, "phi_l": phi_l, "m_lr": m_lr, "phi_lr": phi_lr, "phi_r": phi_r,
        "pack": p["Pack size"].astype(int), "moq": p["Min order qty"].astype(int),
        "unit_cost": p["Unit cost"].astype(float), "price": p["Price"].astype(float), "lead": a.lead,
        "review": a.review, "offset": a.offset, "keep": keep_units(a, keep_one)})
    lam = shadow(a, target, keep_one) if lam is None else lam
    today = items["offset"].to_numpy() == 0
    sup_min = a.shop.suppliers.set_index("Supplier")["Min order value"].astype(float).to_dict()
    out = items.copy()
    for col in ("qty", "value", "last_unit_chance", "fill_without", "fill_with", "risk_before_arrival"):
        out[col] = 0.0
    out["note"] = ""
    if today.any():
        res = policy.plan_orders(items[today].reset_index(drop=True), lam, budget,
                                 {k: v for k, v in sup_min.items() if k in set(items.loc[today, "supplier"])})
        cols = ["qty", "value", "last_unit_chance", "fill_without", "fill_with", "risk_before_arrival", "note"]
        out.loc[today, cols] = res[cols].to_numpy()
    later = ~today
    if later.any():                                 # risk of running out before the next order arrives
        d = a.offset[later] + a.lead[later]
        c = np.concatenate([np.zeros((a.n, 1)), np.cumsum(a.mu, axis=1)], axis=1)
        m_until = c[np.flatnonzero(later), np.minimum(d, a.H)]
        ph = a.disp.phi(m_until / np.maximum(d, 1), np.maximum(d, 1))
        risk = 1 - nb(np.maximum(m_until, 1e-9), ph).cdf(items.loc[later, "position"].to_numpy())
        out.loc[later, "risk_before_arrival"] = np.where(m_until > 1e-9, risk, 0)
        out.loc[later, "fill_without"] = policy.cycle_fill(items.loc[later, "position"].to_numpy(), 0.0, m_l[later],
                                                           phi_l[later], m_lr[later], phi_lr[later], phi_r[later])
        out.loc[later, "note"] = "next order on " + (a.shop.today + pd.to_timedelta(a.offset[later], unit="D")) \
            .strftime("%a %d %b").to_numpy()
    out["today"] = today
    out["qty"] = out["qty"].astype(int)
    out.attrs["lam"] = lam
    return out


def what_if_today(a: Analysis, i: int, lam: float, keep_one=True) -> int:
    """The order for product i if its supplier took orders today."""
    m_l, phi_l, m_lr, phi_lr, phi_r = demand_ahead(a, [i])
    p = a.shop.products.iloc[i]
    pos = float(p["On hand"]) + float(p["On order"])
    P = policy.sell_probs([pos], m_l, phi_l, m_lr, phi_lr, phi_r)
    keep = keep_units(a, keep_one)[i]
    return int(policy.decide(P, [float(p["Price"])], [float(p["Unit cost"])], lam, [int(p["Pack size"])],
                             [int(p["Min order qty"])], [int(max(np.ceil(keep - pos), 0))])[0])


def futures(a: Analysis, i: int, qty: float, size=60, seed=0, days=None):
    """Possible futures of product i's shelf: daily sales drawn around the forecast (with the measured ranges), the
    stock on the shelf day by day with today's order arriving after the lead time. (size, days+1) stock paths."""
    rng = np.random.default_rng(seed)
    L, R = int(a.lead[i]), int(a.review[i])
    days = days or min(a.H, L + R)
    mu = a.mu[i, :days]
    per = mu.mean()
    sales = sample_paths(mu, a.disp.phi(per, 1), a.disp.phi(per, days), size, rng)
    p = a.shop.products.iloc[i]
    stock = np.zeros((size, days + 1))
    stock[:, 0] = float(p["On hand"])
    oh = np.full(size, float(p["On hand"]))         # what is on order is taken to arrive with today's order
    lost = np.zeros(size)
    for d in range(days):
        if d == L:
            oh = oh + qty + float(p["On order"])
        s = np.minimum(oh, sales[:, d])
        lost += sales[:, d] - s
        oh = oh - s
        stock[:, d + 1] = oh
    return {"stock": stock, "sales": sales, "lost": lost, "lead": L, "review": R, "days": days, "mu": mu}


def replay(a: Analysis, low_days=14, high_days=28, target=0.95, weeks=13, keep_one=True, progress=None):
    """The last `weeks` weeks replayed with real sales, ordering two ways: the planner (with a model trained only on
    sales before those weeks) and the rule of thumb (stock between low and high days of average sales)."""
    T, H, n = a.T, a.H, a.n
    weeks = min(int(weeks), (T - 2 * H - 7 - 90) // 7)  # keep at least 90 days to learn from before the replay
    if weeks < 4:
        raise ValueError("the replay needs more sales history (about 4 months plus 4 weeks to replay)")
    D = 7 * weeks
    s0 = T - D                                       # first replayed day
    say = progress or (lambda *_: None)
    say(0.1, "Learning from the sales before the replayed weeks")
    first = max(H + 7, s0 - 1 - H - 730)
    mod = engine.fit(a.hist, H, first_day=first, last_day=s0 - 1 - H, rounds=_rounds(n * 730))
    hm = engine.predict(mod, a.hist, np.arange(s0 - H, s0))
    disp = Dispersion.fit(hm, a.Y[:, s0 - H:s0])
    extra = int((a.lead + a.review).max())
    mu = engine.predict(mod, a.hist, np.arange(s0, min(T + extra, T + H)))
    if mu.shape[1] < D + extra:                      # pad with the last week's pattern
        k = D + extra - mu.shape[1]
        mu = np.concatenate([mu, np.tile(mu[:, -7:], (1, k // 7 + 1))[:, :k]], axis=1)
    p = a.shop.products
    pack, moq = p["Pack size"].to_numpy(float), p["Min order qty"].to_numpy(float)
    cost, price = p["Unit cost"].to_numpy(float), p["Price"].to_numpy(float)
    off = (a.offset - s0 + T) % a.review                # same weekdays as the real order days
    rv = simulate.review_mask(n, D, a.review, off)
    say(0.6, "Replaying")
    keep = (a.Y[:, s0 - 90:s0].sum(axis=1) > 0).astype(float) if keep_one else np.zeros(n)
    w_ = min(28, H)
    f0 = simulate.level_factor(a.Y[:, s0 - w_:s0], hm[:, -w_:])
    lam = simulate.calibrate(mu * f0[:, None], disp, a.lead, a.review, price, cost, target, reps=_reps(n), pack=pack,
                             moq=moq, keep=keep, offset=off)
    recent = (np.concatenate([a.Y[:, s0 - w_:s0], a.Y[:, s0:T]], axis=1),
              np.concatenate([hm[:, -w_:], mu[:, :D]], axis=1), w_)
    plan = simulate.Planner(mu, disp, a.lead, a.review, lam, price, cost, pack, moq, keep, recent)
    S0 = plan.start()
    demand = a.Y[:, s0:T]
    r_plan = simulate.run(demand, a.lead, rv, plan, S0)
    rule = simulate.CoverRule(a.Y[:, :s0], demand, low_days, high_days, pack, moq)
    start_rule = np.ceil(high_days * rule.avg(0, np.arange(n)))
    r_rule = simulate.run(demand, a.lead, rv, rule, start_rule)
    say(1.0, "Done")
    return {"start": a.start + pd.Timedelta(days=s0), "days": D, "plan": r_plan, "rule": r_rule, "lam": lam,
            "kpi_plan": r_plan.kpis(cost, price), "kpi_rule": r_rule.kpis(cost, price),
            "level": S0, "rule_high": start_rule, "cost": cost, "price": price}


def frontier(a: Analysis, points=10, keep_one=True):
    """Average stock value and share of demand value served for a range of shadow prices, from simulating the coming
    weeks over possible futures (the same simulation that sets the target)."""
    p = a.shop.products
    mu = _ahead(a)
    n = a.n
    reps = _reps(n)
    days = max(3 * int((a.lead + a.review).max()), 28)
    days = min(days, mu.shape[1] - int((a.lead + a.review).max()))
    rng = np.random.default_rng(0)
    dem = simulate.sample_demand(mu, a.disp, days, reps, rng)
    t = lambda v: np.tile(np.asarray(v), reps)                                # noqa: E731
    w, c = p["Price"].to_numpy(float), p["Unit cost"].to_numpy(float)
    keep = keep_units(a, keep_one)
    rv = simulate.review_mask(n * reps, days, t(a.review), t(a.offset))
    tot = float((dem.sum(axis=1) * t(w)).sum())
    rows = []
    hi = float(np.max(w / np.maximum(c, 1e-9)))
    for lam in np.geomspace(hi * 1e-3, hi * 0.6, points):
        pl = simulate.Planner(np.tile(mu, (reps, 1)), a.disp, t(a.lead), t(a.review), lam, t(w), t(c),
                              t(p["Pack size"].to_numpy(float)), t(p["Min order qty"].to_numpy(float)), t(keep))
        r = simulate.run(dem, t(a.lead), rv, pl, pl.start())
        rows.append((float((r.sold.sum(axis=1) * t(w)).sum()) / max(tot, 1e-9),
                     float((r.on_hand.mean(axis=1) * t(c)).sum()) / reps, float(lam)))
    df = pd.DataFrame(rows, columns=["target", "stock_value", "lam"]).sort_values("target")
    return df.drop_duplicates("target").reset_index(drop=True)
