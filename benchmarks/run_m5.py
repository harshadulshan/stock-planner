"""
M5 benchmark (Walmart, 30,490 product-store series), in stages:

    python benchmarks/run_m5.py forecast     # train the engine per store at three origins (about 1.5 hours, 2 cores)
    python benchmarks/run_m5.py score        # accuracy, ranges and the stock replay; writes results_m5.json

Origins: the public leaderboard period (train on days 1-1913, forecast 1914-1941) is used to choose the number of
trees, to measure how far sales stray from the forecast (the ranges) and to choose how much ordering corrects the
forecast with the latest weeks' sales (with an earlier origin, train on 1-1885, forecast 1886-1913, for the latest
weeks before it); the official final test (train on 1-1941, forecast 1942-1969) is scored once, with the official
weights, and compared with the published results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stockplan import engine, m5  # noqa: E402

RUNS = os.path.join(ROOT, "data_cache", "m5_runs")
PERIODS = {"early": ("validation", 1885), "validation": ("validation", None), "evaluation": ("evaluation", None)}
CHECKPOINTS = (100, 200, 300, 400, 600)
START = pd.Timestamp("2011-01-29")


def store_history(ids, train, P, cal, store):
    m = (ids["store_id"] == store).to_numpy()
    ev = cal[["event_name_1", "event_type_1"]].rename(columns={"event_name_1": "event", "event_type_1": "event_type"})
    snap = cal[f"snap_{store[:2]}"].to_numpy()
    h = engine.History(train[m], START, ids.loc[m, ["item_id", "dept_id", "cat_id"]].reset_index(drop=True),
                       P[m], ev, np.broadcast_to(snap[None, :], (int(m.sum()), len(snap))))
    return m, h


def load_period(period):
    """(ids, train, test) for an origin; 'early' ends 28 days before the public leaderboard period."""
    src, cut = PERIODS[period]
    ids, train, test = m5.load(src)
    if cut is not None:
        train, test = train[:, :cut], train[:, cut:cut + 28]
    return ids, train, test


def forecast():
    os.makedirs(RUNS, exist_ok=True)
    cal = m5.calendar()
    for period in ("early", "validation", "evaluation"):
        ids, train, test = load_period(period)
        P = m5.prices(ids, len(cal))
        T = train.shape[1]
        for store in sorted(ids["store_id"].unique()):
            out = os.path.join(RUNS, f"{period}_{store}.npz")
            if os.path.exists(out):
                continue
            t0 = time.time()
            m, h = store_history(ids, train, P, cal, store)
            mod = engine.fit(h, 28, rounds=max(CHECKPOINTS))
            days = np.arange(T, T + 28)
            preds = {f"r{r}": engine.predict(mod, h, days, rounds=r).astype(np.float32) for r in CHECKPOINTS}
            np.savez_compressed(out, rows=np.flatnonzero(m), **preds)
            mod.booster.save_model(os.path.join(RUNS, f"{period}_{store}.txt"))
            print(f"{period} {store}: {time.time() - t0:.0f} s", flush=True)


def stock_at(curve, fill):
    """Days of stock a curve (points of service and stock) needs for `fill`, inf outside its range."""
    f = np.array([p["value_fill"] for p in curve])
    s = np.array([p["stock_days"] for p in curve])
    o = np.argsort(f)
    f, s = f[o], s[o]
    return float(np.interp(fill, f, s)) if f.min() <= fill <= f.max() else float("inf")


def same_service(curve, others, key):
    """For each point of `others` inside the curve's range of service: [its setting, service, its stock in days, the
    curve's stock for that service, the share of stock the curve saves (negative: it needs more)]."""
    res = []
    for o in others:
        need = stock_at(curve, o["value_fill"])
        if np.isfinite(need):
            res.append([o[key], round(o["value_fill"], 4), round(o["stock_days"], 3), round(need, 3),
                        round(1 - need / o["stock_days"], 4)])
    return res


def _phrase(comps, same=0.03):
    """'about the same stock at 86.9% and 92.8% served (2% and 1% more), 5% less at 95.6% ...'"""
    groups = []
    for c in comps:
        kind = "same" if abs(c[4]) < same else ("less" if c[4] > 0 else "more")
        if groups and groups[-1][0] == kind == "same":
            groups[-1][1].append(c)
        else:
            groups.append((kind, [c]))
    parts = []
    for kind, cs in groups:
        at = " and ".join(f"{c[1]:.1%}" for c in cs) + " served"
        if kind == "same":
            parts.append(f"about the same stock at {at} (" + " and ".join(
                f"{abs(c[4]):.0%}" for c in cs) + f" {'less' if cs[0][4] >= 0 else 'more'})")
        else:
            parts.append(f"{abs(cs[0][4]):.0%} {kind} at {at}")
    return ", ".join(parts)


def inventory_summary(out):
    """Comparisons and the summary text of the ordering test, from its curves."""
    inv = out["inventory"]
    pl = inv["planner"]
    inv["same_fill"] = same_service(pl, inv["rule"], "cover")
    inv["same_fill_without"] = same_service(inv["without_correction"], inv["rule"], "cover")
    inv["per_product_vs"] = same_service(pl, inv["per_product"], "target")
    inv["correction_saves"] = same_service(pl, inv["without_correction"], "target")
    t95 = [p for p in pl if abs(p["target"] - 0.95) < 1e-9][0]
    cs = [c[4] for c in inv["correction_saves"]]
    txt = []
    if inv["same_fill"]:
        txt.append("For the same service as the rule of thumb, Stock Planner needs " + _phrase(inv["same_fill"]) + ".")
    if cs:
        txt.append(f"Bringing in the latest weeks' sales (weight {inv['correction_weight']:g}, chosen on the earlier "
                   f"period) saves {min(cs):.0%} to {max(cs):.0%} of the stock for the same service.")
    pp = [c[4] for c in inv["per_product_vs"]]
    if pp:
        txt.append("Giving every product the same target instead of one shop-wide rule needs about the same stock "
                   f"(within {max(abs(x) for x in pp):.0%})." if max(abs(x) for x in pp) < 0.03 else
                   "Against the same target for every product: " + _phrase(inv["per_product_vs"]) + ".")
    txt.append(f"Asked for 95% of the demand value, the orders served {t95['value_fill']:.1%} on the final test (the "
               "target is set by simulating possible futures around the forecast; the real weeks were harder).")
    inv["summary"] = " ".join(txt)
    lw = out.get("latest_weeks")
    if lw:
        lw["summary"] = ("On the earlier period, for the same service as the rule of thumb: "
                         + _phrase(lw["same_fill"]) + ".") if lw["same_fill"] else ""
    return out


def load_forecasts(period, n, rounds):
    F = np.zeros((n, 28), np.float32)
    for f in sorted(os.listdir(RUNS)):
        if f.startswith(period + "_") and f.endswith(".npz"):
            z = np.load(os.path.join(RUNS, f), allow_pickle=False)
            F[z["rows"]] = z[f"r{rounds}"]
    return F


def score():
    from stockplan import policy, simulate
    from stockplan.uncertainty import Dispersion, quantiles

    t0 = time.time()
    pub = json.load(open(os.path.join(ROOT, "benchmarks", "m5_published.json")))
    out = {"machine": f"{os.cpu_count()} cores", "checkpoints": list(CHECKPOINTS)}
    # 1. number of trees, chosen on the public leaderboard period
    ids_v, tr_v, te_v = m5.load("validation")
    h_v = m5.Hierarchy(ids_v, "validation")
    val = {r: m5.wrmsse(h_v, tr_v, te_v, load_forecasts("validation", len(ids_v), r))["WRMSSE"] for r in CHECKPOINTS}
    best = min(val, key=val.get)
    out["validation"] = {"wrmsse_by_trees": {str(k): round(v, 5) for k, v in val.items()}, "trees": best,
                         "snaive": round(m5.wrmsse(h_v, tr_v, te_v, m5.snaive(tr_v))["WRMSSE"], 5)}
    print("validation", out["validation"], flush=True)
    # 2. the official final test, scored once
    ids, tr, te = m5.load("evaluation")
    h = m5.Hierarchy(ids, "evaluation")
    F = load_forecasts("evaluation", len(ids), best).astype(np.float64)
    r = m5.wrmsse(h, tr, te, F)
    top = pub["accuracy_top50"]
    better = sum(1 for _, v in top if v < r["WRMSSE"])
    rank_text = (f"would place {better + 1} of {pub['teams_accuracy']:,} teams" if better < len(top)
                 else f"outside the top 50 (50th: {top[-1][1]:.3f})")
    bench = {k: v["WRMSSE"] for k, v in pub["accuracy_benchmarks"].items()}
    out["accuracy"] = {"engine": round(r["WRMSSE"], 5), "levels": {k: round(v, 5) for k, v in r.items()
                                                                   if k != "WRMSSE"},
                       "benchmarks": bench, "best_benchmark": min(bench, key=bench.get), "top": top,
                       "rank_text": rank_text, "teams": pub["teams_accuracy"],
                       "check_snaive": round(m5.wrmsse(h, tr, te, m5.snaive(tr))["WRMSSE"], 6)}
    print("accuracy", out["accuracy"]["engine"], rank_text, flush=True)
    # 3. ranges: dispersion measured on the public period, scored on the final test (level 12)
    Fv = load_forecasts("validation", len(ids_v), best).astype(np.float64)
    disp = Dispersion.fit(Fv, te_v)
    per = F.mean(axis=1, keepdims=True) * np.ones_like(F)
    qs = quantiles(F, disp.phi(per, 1), m5.QUANTILES)
    spl = m5.spl_bottom(h, tr, te, qs)
    cover = {}
    for lo, hi in ((0.25, 0.75), (0.165, 0.835), (0.025, 0.975), (0.005, 0.995)):
        cover[f"{hi - lo:.0%}"] = round(float(((te >= qs[lo]) & (te <= qs[hi])).mean()), 4)
    ub = pub["uncertainty_benchmarks"]
    out["uncertainty"] = {
        "engine": round(spl["mean"], 5), "by_quantile": {str(k): round(v, 5) for k, v in spl.items() if k != "mean"},
        "benchmarks": {k: v["L12"] for k, v in ub.items()}, "top_l12": [[t, v12] for t, _, v12 in
                                                                         pub["uncertainty_top50"]],
        "coverage": cover, "dispersion": disp.to_dict(),
        "note": ("Level 12 (each product in each store) of the M5 uncertainty competition: 9 quantiles a day, scaled "
                 "pinball loss with the official weights. The other levels need ranges for sums of products, which "
                 "the shop app does not use. Coverage of the central ranges on the final test: "
                 + ", ".join(f"{k} range holds {v:.0%} of the days" for k, v in cover.items()) + ".")}
    print("uncertainty", out["uncertainty"]["engine"], cover, flush=True)
    # 4. ordering with the forecasts: every product in every store, 28 days, weekly orders, 3 days lead time
    L, R = 3, 7
    n = len(ids)
    P = m5.prices(ids, len(m5.calendar()))
    lead, rvd = np.full(n, L), np.full(n, R)
    rv = simulate.review_mask(n, 28, R)

    def shelf_price(day):
        p = P[:, day]
        last = pd.DataFrame(P[:, :day]).ffill(axis=1).iloc[:, -1].to_numpy()
        p = np.where(np.isnan(p), last, p)
        return np.nan_to_num(p, nan=float(np.nanmedian(p)))

    def kp(res, dem, price):
        k = res.kpis(price, price)
        sold, d_ = res.sold.sum(1), dem.sum(1)
        day_value = float((dem * price[:, None]).sum() / dem.shape[1])
        return {"value_fill": round(k["value_fill"], 5), "fill_rate": round(k["fill_rate"], 5),
                "stock_days": round(k["avg_stock_value"] / day_value, 3),
                "item_fill_median": round(float(np.median(sold[d_ > 0] / d_[d_ > 0])), 4),
                "stockout_share": round(float((res.lost.sum(1) > 0).mean()), 4)}

    def rule_curve(hist, dem, price):
        res_ = []
        for c in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
            days = c * (L + R)
            cr = simulate.CoverRule(hist[:, -28:], dem, days, days)
            res = simulate.run(dem, lead, rv, cr, np.ceil(days * cr.avg(0, np.arange(n))))
            res_.append({"cover": c, **kp(res, dem, price)})
        return res_

    # 4a. the public leaderboard period: how much ordering should correct the forecast with the latest weeks' sales.
    # The forecast's inputs are at least 28 days old; the error of the forecasts made on day 1885 for days 1886-1913
    # says how far each product has drifted lately (the final test is not used for this choice).
    _, _, te_e = load_period("early")
    Fe = load_forecasts("early", n, best).astype(np.float64)
    disp_v = Dispersion.fit(Fe, te_e)
    price_v = shelf_price(tr_v.shape[1])
    mu_v = np.concatenate([Fv, Fv[:, 21:24]], axis=1)
    recent_v = (np.concatenate([tr_v[:, -28:], te_v], axis=1), np.concatenate([Fe, Fv], axis=1), 28)
    FILLS = (0.90, 0.95, 0.97)
    weights = {}
    for wgt in (0.0, 0.25, 0.5, 0.75, 1.0):
        curve = []
        for lam in (0.3, 0.15, 0.08, 0.04, 0.02):
            pl = simulate.Planner(mu_v, disp_v, lead, rvd, lam, price_v, price_v, recent=recent_v, level_weight=wgt)
            curve.append({"lam": lam, **kp(simulate.run(te_v, lead, rv, pl, pl.start()), te_v, price_v)})
        need = [stock_at(curve, f) for f in FILLS]                       # inf: the curve does not reach it
        weights[wgt] = {"curve": curve, "stock_days_at": [round(x, 3) if np.isfinite(x) else None for x in need],
                        "mean": float(np.mean(need))}
        print("correction weight", wgt, weights[wgt]["stock_days_at"], round(time.time() - t0), flush=True)
    wbest = min(weights, key=lambda k: weights[k]["mean"])
    for v in weights.values():
        v.pop("mean")
    rule_v = rule_curve(tr_v, te_v, price_v)
    out["latest_weeks"] = {
        "period": "public leaderboard period (days 1914-1941): forecasts made on day 1913, ranges and the latest "
                  "weeks' error from forecasts made on day 1885",
        "fills": list(FILLS), "weights": {str(k): v for k, v in weights.items()}, "chosen": wbest,
        "rule": rule_v, "same_fill": same_service(weights[wbest]["curve"], rule_v, "cover"),
        "same_fill_without": same_service(weights[0.0]["curve"], rule_v, "cover")}
    print("chosen weight", wbest, out["latest_weeks"]["same_fill"], flush=True)
    # 4b. the final test, ordered with the weight chosen above
    price = shelf_price(tr.shape[1])
    mu = np.concatenate([F, F[:, 21:24]], axis=1)                 # days 29-31: same weekday a week earlier
    recent = (np.concatenate([tr[:, -28:], te], axis=1), np.concatenate([Fv, F], axis=1), 28)
    f0 = simulate.level_factor(tr[:, -28:], Fv, wbest)
    planner, without = [], []
    for tg in (0.85, 0.9, 0.95, 0.97, 0.98, 0.99):
        lam = simulate.calibrate(mu * f0[:, None], disp, lead, rvd, price, price, tg)
        pl = simulate.Planner(mu, disp, lead, rvd, lam, price, price, recent=recent, level_weight=wbest)
        planner.append({"target": tg, "lam": round(lam, 5), **kp(simulate.run(te, lead, rv, pl, pl.start()), te, price)})
        p0 = simulate.Planner(mu, disp, lead, rvd, lam, price, price)        # the same without the correction
        without.append({"target": tg, "lam": round(lam, 5), **kp(simulate.run(te, lead, rv, p0, p0.start()), te,
                                                                    price)})
        print("shop-wide", planner[-1], "| without", without[-1]["value_fill"], without[-1]["stock_days"],
              round(time.time() - t0), flush=True)

    class PerProduct(simulate.Planner):
        """The same service for every product: order up to its own fill-rate target (for comparison)."""
        def __init__(self, target, *a_, **k_):
            super().__init__(*a_, **k_)
            self.target = target

        def __call__(self, d, rows, on_hand, position):
            m_l, phi_l, m_lr, phi_lr, phi_r = self.demand_ahead(d, rows)
            return policy.order_qty(position, m_l, phi_l, m_lr, phi_lr, phi_r, np.full(len(rows), self.target))

    per_product = []
    for tg in (0.85, 0.9, 0.95, 0.99):
        pp = PerProduct(tg, mu, disp, lead, rvd, 0.0, price, price, recent=recent, level_weight=wbest)
        m_l, phi_l, m_lr, phi_lr, _ = pp.demand_ahead(0, np.arange(n))
        S0 = policy.order_up_to(m_lr, phi_lr, m_l, phi_l, np.full(n, tg))
        per_product.append({"target": tg, **kp(simulate.run(te, lead, rv, pp, S0), te, price)})
        print("per product", per_product[-1], round(time.time() - t0), flush=True)
    rule = rule_curve(tr, te, price)
    for rr in rule:
        print("rule", rr, flush=True)
    out["inventory"] = {
        "planner": planner, "without_correction": without, "per_product": per_product, "rule": rule,
        "correction_weight": wbest, "lead_days": L, "review_days": R,
        "setup": (f"Every product in every store ({n:,}), the 28 days of the final test. Orders every {R} days, "
                  f"delivered {L} days later; actual sales are the demand and an empty shelf loses the sale. Stock "
                  "Planner orders from the forecasts made before the 28 days, corrected with the latest weeks' sales "
                  "(how much was chosen on the earlier period); the rule of thumb orders up to a multiple of 10 days "
                  "of average sales (last 28 days); 'same target for every product' orders each product up to its "
                  "own fill-rate target. All start from their own full level. Service is the share of demand value "
                  "served; stock is valued at shelf price and shown in days of sales.")}
    inventory_summary(out)
    print(out["inventory"]["summary"], flush=True)
    out["seconds_score"] = round(time.time() - t0)
    json.dump(out, open(os.path.join(ROOT, "benchmarks", "results_m5.json"), "w"), indent=1)
    print("done", round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["forecast", "score", "summary"])
    a = ap.parse_args()
    if a.stage == "forecast":
        forecast()
    elif a.stage == "score":
        score()
    else:                                   # comparisons and texts again, from the saved numbers
        path = os.path.join(ROOT, "benchmarks", "results_m5.json")
        out = inventory_summary(json.load(open(path)))
        json.dump(out, open(path, "w"), indent=1)
        print(out["inventory"]["summary"])


if __name__ == "__main__":
    main()
