"""
Engine checks: the M5 scorer against the published scores, forecasts that never look past the decision day, demand
ranges, order quantities against exact computation and simulation, budgets and supplier minimums, the replay, the
Excel shop format and the demo.

    python tests/verify_engine.py            # about 3 minutes
    python tests/verify_engine.py --quick    # without the M5 data checks
"""

from __future__ import annotations

import os
import sys

ROOT = os.environ.get("SP_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stockplan import analysis, engine, m5, policy, simulate  # noqa: E402
from stockplan import shop as shopmod  # noqa: E402
from stockplan.uncertainty import Dispersion, nb, quantiles, sample_paths  # noqa: E402

QUICK = "--quick" in sys.argv
passed, failed = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


rng = np.random.default_rng(11)

print("M5 accuracy measure")
y = np.array([[0, 0, 3, 1, 4, 0, 2.0], [1, 2, 3, 4, 5, 6, 7.0]])
d = np.diff(y, axis=1)
want = np.array([np.mean(d[0, 2:] ** 2), np.mean(d[1] ** 2)])
ok("scale: mean squared day-to-day change from the first sale on", np.allclose(m5.scales(y), want),
   f"{m5.scales(y)} vs {want}")
ok("absolute scale for the pinball loss", np.allclose(m5.scales(y, squared=False),
                                                      [np.mean(np.abs(d[0, 2:])), np.mean(np.abs(d[1]))]))
have_m5 = all(os.path.exists(os.path.join(m5.CACHE, f)) for f in m5.FILES)
if have_m5:
    ids, tr, te = m5.load("evaluation")
    h = m5.Hierarchy(ids, "evaluation")
    ok("42,840 series over 12 levels, each level's weights sum to 1",
       h.A.shape == (42840, 30490) and all(abs(h.weight[h.level_of == lv].sum() - 1) < 1e-6 for lv, _ in m5.LEVELS))
    s_n = m5.wrmsse(h, tr, te, m5.snaive(tr))["WRMSSE"]
    ok("seasonal naive scores the published 0.847017", abs(s_n - 0.847017) < 2e-6, f"{s_n:.6f}")
    s_v = m5.wrmsse(h, tr, te, m5.naive(tr))["WRMSSE"]
    ok("naive scores the published 1.752010", abs(s_v - 1.752010) < 2e-6, f"{s_v:.6f}")
    qs = {0.5: m5.snaive(tr), 0.9: m5.snaive(tr) + 1}
    spl = m5.spl_bottom(h, tr, te, qs)
    lv = h.level_of == "Level12"
    w_bottom = np.zeros(len(ids))
    w_bottom[np.asarray(h.A[lv].argmax(axis=1)).ravel()] = h.weight[lv]
    sc = m5.scales(tr, squared=False)
    e = te - qs[0.9]
    pin = np.where(e >= 0, 0.9 * e, -0.1 * e).mean(axis=1)
    manual = float((w_bottom * pin / np.maximum(sc, 1e-12)).sum())
    ok("level-12 pinball loss matches a direct computation", abs(spl[0.9] - manual) < 1e-9, f"{spl[0.9]} {manual}")
else:
    print("  (M5 data not downloaded: skipping the published-score checks)")

print("Forecast engine")
T = 400
t_ = np.arange(T)
base = 40 * (1 + 0.6 * (t_ % 7 == 5) - 0.3 * (t_ % 7 == 6)) * (1 + 0.3 * np.sin(2 * np.pi * t_ / 365))
Y = rng.poisson(np.vstack([base * f for f in (1.0, 0.5, 2.0, 0.2)])).astype(np.float32)
hist = engine.History(Y, pd.Timestamp("2024-01-01"), pd.DataFrame({"cat": ["a", "a", "b", "b"]}))
days = np.arange(T - 30, T + 10)
for H in (10, 14):
    f1, _, _ = engine.features(hist, H, days)
    Y2 = Y.copy()
    cut = T - 30 - H                                  # sales after this day must not matter for day T-30
    Y2[:, cut + 1:] = rng.poisson(50, size=Y2[:, cut + 1:].shape)
    f2, _, _ = engine.features(engine.History(Y2, hist.start, hist.static), H, days[:1])
    same = all(np.allclose(np.nan_to_num(f1[k][:, :1]), np.nan_to_num(f2[k])) for k in f1)
    ok(f"features for a day use only sales at least {H} days older", same,
       [k for k in f1 if not np.allclose(np.nan_to_num(f1[k][:, :1]), np.nan_to_num(f2[k]))][:3])
try:
    engine.features(hist, 10, np.array([T + 10]))
    ok("a day beyond the horizon is refused", False)
except ValueError:
    ok("a day beyond the horizon is refused", True)
mod = engine.fit(hist, 14, rounds=120)
pred = engine.predict(mod, hist, np.arange(T - 14, T))
mod2 = engine.fit(hist, 14, last_day=T - 15, rounds=120)
held = engine.predict(mod2, hist, np.arange(T - 14, T))
actual = Y[:, T - 14:]
naive = np.repeat(Y[:, T - 42:T - 14].mean(axis=1, keepdims=True), 14, axis=1)
err = np.abs(held - actual).sum() / actual.sum()
err_naive = np.abs(naive - actual).sum() / actual.sum()
ok("learns a weekly pattern: better than the 4-week average on unseen days", err < err_naive * 0.95,
   f"{err:.3f} vs {err_naive:.3f}")
ok("forecasts are never negative", (pred >= 0).all() and (held >= 0).all())

print("Demand ranges")
mu = np.full((3000, 28), 4.0)
true_phi = 0.3
sales = rng.poisson(rng.gamma(1 / true_phi, mu * true_phi))
dsp = Dispersion.fit(mu, sales)
p1, p7 = dsp.overall[0], dsp.overall[1]
ok("daily dispersion recovered", abs(p1 - true_phi) < 0.03, f"{p1:.3f}")
ok("weekly dispersion of independent days is about 1/7 of the daily", abs(p7 - true_phi / 7) < 0.015, f"{p7:.4f}")
q = quantiles(np.array([3.0, 30.0]), np.array([0.2, 0.2]), [0.05, 0.5, 0.95])
ok("quantiles increase", (q[0.05] <= q[0.5]).all() and (q[0.5] <= q[0.95]).all())
paths = sample_paths(np.full(14, 5.0), 0.2, 0.08, 20000, rng)
tot = paths.sum(axis=1)
M = 70.0
ok("possible futures: mean and the measured range of the total", abs(tot.mean() / M - 1) < 0.02 and
   abs(tot.var() / (M + 0.08 * M * M) - 1) < 0.08, f"{tot.mean():.1f} {tot.var():.0f} vs {M + 0.08 * M * M:.0f}")

print("Order quantities")
n = 300
m_l = rng.gamma(1, 3, n) * (rng.random(n) > 0.1)
m_r = rng.gamma(1, 6, n)
m_lr = m_l + m_r
phi_l = rng.random(n) * 0.5
phi_lr = phi_l + rng.random(n) * 0.3
tg = rng.choice([0.8, 0.9, 0.95, 0.99], n)
S = policy.order_up_to(m_lr, phi_lr, m_l, phi_l, tg)
fS = policy.fill_rate(S, m_lr, phi_lr, m_l, phi_l)
fS1 = policy.fill_rate(np.maximum(S - 1, 0), m_lr, phi_lr, m_l, phi_l)
ok("order-up-to level: the smallest that meets the target (300 cases)",
   ((fS >= tg - 1e-9) & ((S == 0) | (fS1 < tg))).all())
cases = [(2, 3.0, 0.2, 7.0, 0.1, 0.8), (0, 3.0, 0.2, 7.0, 0.1, 0.9), (5, 6.0, 0.4, 10.0, 0.2, 0.95),
         (1, 2.0, 0.1, 4.0, 0.05, 0.8)]
bad = []
for A, ml, pl, mr, pr, t in cases:
    q_ = policy.order_qty([A], [ml], [pl], [ml + mr], [pl], [pr], [t])[0]
    DL = rng.poisson(rng.gamma(1 / pl, ml * pl, 200000))
    DR = rng.poisson(rng.gamma(1 / pr, mr * pr, 200000))

    def mc(qq):
        stock = np.maximum(A - DL, 0) + qq
        return 1 - np.maximum(DR - stock, 0).mean() / mr
    if not (mc(q_) >= t - 0.004 and (q_ == 0 or mc(q_ - 1) < t + 0.004)):
        bad.append((A, t, q_, round(mc(q_), 4), round(mc(q_ - 1), 4) if q_ else None))
ok("lost-sales order quantity: smallest that meets the target (exact vs simulation)", not bad, str(bad))
ok("whole packs and minimum order quantity", list(policy.round_order([0, 1, 13, 25, 3], [6, 6, 6, 6, 1],
                                                                     [0, 12, 12, 12, 5])) == [0, 12, 18, 30, 5])

print("Shop-wide ordering")
n = 150
A_ = rng.integers(0, 25, n)
m_l2 = rng.gamma(1, 4, n) * (rng.random(n) > .1)
m_r2 = rng.gamma(1, 8, n)
phl, phr = rng.random(n) * .5, rng.random(n) * .4
phlr = phl * .6 + .05
P = policy.sell_probs(A_, m_l2, phl, m_l2 + m_r2, phlr, phr)
worst = 0.0
for i in range(n):
    K = P.shape[1]
    kk = np.arange(1, K + 1)
    risky = m_l2[i] > 1e-9 and nb(m_l2[i], phl[i]).sf(A_[i] - 1) > 1e-6
    if not risky:
        ref = nb(m_l2[i] + m_r2[i], phlr[i]).sf(A_[i] + kk - 1) - (nb(m_l2[i], phl[i]).sf(A_[i] + kk - 1)
                                                                    if m_l2[i] > 1e-9 else 0)
        ref = np.clip(ref, 0, 1)
    else:
        x = np.arange(A_[i] + 1)
        pX = np.where(x == 0, nb(m_l2[i], phl[i]).sf(A_[i] - 1), nb(m_l2[i], phl[i]).pmf(A_[i] - x))
        ref = np.array([(pX * nb(m_r2[i], phr[i]).sf(x + k_ - 1)).sum() for k_ in kk])
    worst = max(worst, float(np.abs(P[i] - ref).max()))
ok("chance that each unit sells: fast tables match the direct formulas", worst < 1e-9, f"{worst:.2e}")
i = int(np.argmax([(m_l2[j] > 0) * nb(max(m_l2[j], 1e-9), phl[j]).sf(A_[j] - 1) * (A_[j] > 0) for j in range(n)]))
DL = rng.poisson(rng.gamma(1 / phl[i], m_l2[i] * phl[i], 400000))
DR = rng.poisson(rng.gamma(1 / phr[i], m_r2[i] * phr[i], 400000))
X = np.maximum(A_[i] - DL, 0)
mcp = np.array([(DR >= X + k_).mean() for k_ in range(1, 6)])
ok("... and a simulation of the shelf when it can run out before the delivery",
   np.abs(mcp - P[i, :5]).max() < 0.004, f"{mcp.round(4)} vs {P[i, :5].round(4)}")
q1 = policy.decide(np.array([[0.9, 0.7, 0.5, 0.3, 0.1, 0.05]] * 3), [10.0] * 3, [4.0] * 3, 1.0, [1, 2, 4],
                   [0, 0, 6])
ok("a unit is ordered while price x chance of selling >= shadow price x cost (packs, minimums)",
   list(q1) == [3, 2, 0], str(q1))
items = pd.DataFrame({"position": [2.0, 1.0, 0.0], "on_hand": [2.0, 1.0, 0.0], "m_l": [3.0, 2.0, 1.0],
                      "phi_l": [0.1] * 3, "m_lr": [12.0, 9.0, 4.0], "phi_lr": [0.1] * 3, "phi_r": [0.1] * 3,
                      "pack": [4, 3, 2], "moq": [0, 0, 0], "unit_cost": [10.0, 20.0, 5.0], "price": [15.0, 40.0, 6.0],
                      "supplier": ["S", "S", "S"], "keep": [0, 0, 0]})
full = policy.plan_orders(items, 0.2)
tight = policy.plan_orders(items, 0.2, budget=60)
ok("with a budget: never spends more", tight["value"].sum() <= 60 + 1e-9 and full["value"].sum() > 60,
   f"{tight['value'].sum()} {full['value'].sum()}")
big = policy.plan_orders(items, 0.2, budget=10 ** 6)
ok("an ample budget changes nothing", (big["qty"] == full["qty"]).all())
Pi = policy.sell_probs(items["position"], items["m_l"], items["phi_l"], items["m_lr"], items["phi_lr"], items["phi_r"])


def sold_value(q):
    return sum(items.at[i, "price"] * Pi[i, :int(q[i])].sum() for i in range(3))


best = 0.0
for a_ in range(0, int(full.at[0, "qty"]) + 1, 4):
    for b_ in range(0, int(full.at[1, "qty"]) + 1, 3):
        for c_ in range(0, int(full.at[2, "qty"]) + 1, 2):
            if 10 * a_ + 20 * b_ + 5 * c_ <= 60:
                best = max(best, sold_value([a_, b_, c_]))
ok("budget spent where it sells most (within 3% of the best combination)",
   sold_value(tight["qty"].to_numpy()) >= 0.97 * best, f"{sold_value(tight['qty'].to_numpy()):.2f} vs {best:.2f}")
calm = items.copy()
calm["position"] = calm["on_hand"] = [20.0, 15.0, 9.0]
waits = policy.plan_orders(calm, 0.2, supplier_min={"S": 10 ** 5})
urgent = policy.plan_orders(items, 0.2, supplier_min={"S": 200})
poor = policy.plan_orders(items, 0.2, budget=150, supplier_min={"S": 200})
ok("below the supplier's minimum: waits when nothing is urgent, topped up when it is, waits if the budget cannot",
   waits["qty"].sum() == 0 and urgent["value"].sum() >= 200 and poor["qty"].sum() == 0,
   f"{waits['qty'].sum()} {urgent['value'].sum()} {poor['value'].sum()}")
kp1 = items.copy()
kp1["position"] = kp1["on_hand"] = [0.0, 0.0, 0.0]
kp1["m_l"], kp1["m_lr"] = 0.001, 0.01
kp1["keep"] = [1, 0, 1]
kq = policy.plan_orders(kp1, 0.5)
ok("one unit of a recently sold product stays on the shelf when asked", list(kq["qty"] > 0) == [True, False, True],
   str(list(kq["qty"])))

print("Replay")
dem = rng.poisson(3, size=(50, 60)).astype(float)
lead = rng.integers(0, 5, 50)
rv = simulate.review_mask(50, 60, 7, rng.integers(0, 7, 50))
log = []


def dec(day, rows, oh, pos):
    q_ = np.full(len(rows), 5.0)
    log.append((day, rows))
    return q_


rp = simulate.run(dem, lead, rv, dec, np.full(50, 10.0))
bal = np.allclose(10 + rp.arrived.sum(axis=1) - rp.sold.sum(axis=1), rp.on_hand[:, -1])
ok("every unit accounted for; sold + lost = demand; never below zero",
   bal and np.allclose(rp.sold + rp.lost, dem) and (rp.on_hand >= -1e-9).all())
arr_ok = all(np.allclose(rp.arrived[i, d + lead[i]], 5.0) for d, rows in log for i in rows if d + lead[i] < 60)
ok("orders arrive exactly after the lead time", arr_ok)
seen = rng.poisson(3, size=(5, 30)).astype(float)
c1 = simulate.CoverRule(np.full((5, 28), 3.0), seen, 7, 14)
seen2 = seen.copy()
seen2[:, 10:] = 99
c2 = simulate.CoverRule(np.full((5, 28), 3.0), seen2, 7, 14)
ok("the rule of thumb uses only sales before the decision day",
   np.allclose(c1(10, np.arange(5), np.zeros(5), np.zeros(5)), c2(10, np.arange(5), np.zeros(5), np.zeros(5))))
disp2 = Dispersion()
disp2.table[:] = np.array([0.3, 0.3 / 7, 0.3 / 14, 0.3 / 28])[:, None]
lf = simulate.level_factor(np.array([[3.0] * 28, [0.0] * 28, [30.0] * 28, [4.0] * 28]), np.full((4, 28), 3.0), 0.5)
ok("latest weeks' correction: none when the forecast was right, part of the way when not, between half and double",
   np.allclose(lf, [1.0, 0.5 + 0.5 * 2 / 86, 2.0, 0.5 + 0.5 * 114 / 86]), str(np.round(lf, 4)))
act1 = np.full((5, 28 + 40), 3.0)
act2 = act1.copy()
act2[:, 28 + 10:] = 99
pa = [simulate.Planner(np.full((5, 40), 3.0), disp2, 3, 7, 0.1, 1.0, 1.0, recent=(x_, np.full((5, 68), 3.0), 28))
      for x_ in (act1, act2)]
m10 = [p_.demand_ahead(10, np.arange(5))[2] for p_ in pa]
m11 = [p_.demand_ahead(11, np.arange(5))[2] for p_ in pa]
ok("the correction uses the sales up to the day before each decision, not the day itself",
   np.allclose(m10[0], m10[1]) and (m11[1] > m11[0] + 1).all(), f"{m10[0][0]:.2f}/{m10[1][0]:.2f} "
                                                                 f"{m11[0][0]:.2f}/{m11[1][0]:.2f}")
nn, DD = 1200, 84
rate = np.exp(rng.normal(-0.5, 1.4, nn))
price = np.exp(rng.normal(3, 1, nn))
demand = rng.poisson(rng.gamma(1 / 0.3, rate[:, None] * 0.3, size=(nn, DD))).astype(float)
mu2 = np.repeat(rate[:, None], DD + 20, axis=1)
day_val = (demand * price[:, None]).sum() / DD
res, curve = [], []
for t in (0.9, 0.95, 0.98, 0.99):
    lam = simulate.calibrate(mu2, disp2, np.full(nn, 3), np.full(nn, 7), price, price, t)
    pl = simulate.Planner(mu2, disp2, np.full(nn, 3), np.full(nn, 7), lam, price, price)
    r_ = simulate.run(demand, np.full(nn, 3), simulate.review_mask(nn, DD, 7), pl, pl.start())
    k_ = r_.kpis(price, price)
    res.append((t, round(k_["value_fill"], 4)))
    curve.append((k_["value_fill"], k_["avg_stock_value"] / day_val))
ok("shop-wide target reached when the demand is known (90, 95, 98, 99%, within 2.5 points)",
   all(-0.005 < a_ - t < 0.025 for t, a_ in res), str(res))
hist = rng.poisson(rng.gamma(1 / 0.3, rate[:, None] * 0.3, size=(nn, 28))).astype(float)
cr = simulate.CoverRule(hist, demand, 15, 15)
rr = simulate.run(demand, np.full(nn, 3), simulate.review_mask(nn, DD, 7), cr, np.ceil(15 * cr.avg(0, np.arange(nn))))
kr_ = rr.kpis(price, price)
cv = sorted(curve)
need = float(np.interp(kr_["value_fill"], [c[0] for c in cv], [c[1] for c in cv]))
ok("same service as the rule of thumb (15 days of cover) with less stock, on mixed slow and fast sellers",
   cv[0][0] <= kr_["value_fill"] <= cv[-1][0] and need < 0.9 * kr_["avg_stock_value"] / day_val,
   f"rule {kr_['value_fill']:.3f} with {kr_['avg_stock_value'] / day_val:.1f} days, planner needs {need:.1f}")

print("Shop data and the demo")
demo = shopmod.demo()
back = shopmod.from_excel(shopmod.to_excel(demo))
Ya, s1 = demo.matrix()
Yb, s2 = back.matrix()
ok("Excel round trip keeps products, suppliers, sales and settings",
   np.array_equal(Ya, Yb) and s1 == s2 and back.today == demo.today and
   back.products.astype(str).equals(demo.products.astype(str)) and len(back.events) == len(demo.events))
tiny = shopmod.Shop("t", "Rs", pd.Timestamp("2026-01-10"),
                    pd.DataFrame([["A", "a", "c", "S", 1, 2, 0, 0, 1, 0]], columns=shopmod.PRODUCT_COLS),
                    pd.DataFrame([["S", 1, 7, pd.Timestamp("2026-01-10"), 0]], columns=shopmod.SUPPLIER_COLS),
                    pd.DataFrame({"Date": ["2026-01-08", "2026-01-09", "2026-01-09"], "SKU": ["A", "A", "A"],
                                  "Qty": [2, 3, 4]}))
Yt, st_ = tiny.matrix()
ok("daily sales matrix: right day, same-day rows added up", Yt.tolist() == [[2.0, 7.0]] and str(st_.date()) ==
   "2026-01-08")
broken = shopmod.demo()
broken.products.loc[0, "Supplier"] = "Nobody Ltd"
broken.products.loc[1, "On hand"] = -4
probs = shopmod.check(broken)
ok("data problems are reported (unknown supplier, negative stock)",
   any("Nobody" in p_ for p_ in probs) and any("On hand" in p_ for p_ in probs), str(probs))
ok("too short a history is reported", any("4 months" in p_ for p_ in shopmod.check(tiny)))
an = analysis.analyse(demo)
acc = an.accuracy
ok("demo: forecasts beat the 4-week average on the weeks they had not seen",
   acc["error_engine"].sum() < acc["error_average"].sum(),
   f"{acc['error_engine'].sum():.0f} vs {acc['error_average'].sum():.0f}")
od = analysis.order_today(an, 0.95)
t_rows = od[od["today"] & (od["qty"] > 0)]
ok("demo orders: whole packs, at least the minimum, only for today's suppliers",
   len(t_rows) > 0 and (t_rows["qty"] % t_rows["pack"] == 0).all() and (t_rows["qty"] >= t_rows["moq"]).all()
   and (od.loc[~od["today"], "qty"] == 0).all())
rp1 = analysis.replay(an, weeks=8)
ok("demo replay: serves more of the demand than the rule of thumb, with no more stock",
   rp1["kpi_plan"]["value_fill"] > rp1["kpi_rule"]["value_fill"] and
   rp1["kpi_plan"]["avg_stock_value"] <= rp1["kpi_rule"]["avg_stock_value"],
   f"{rp1['kpi_plan']['value_fill']:.3f}/{rp1['kpi_rule']['value_fill']:.3f} "
   f"{rp1['kpi_plan']['avg_stock_value']:.0f}/{rp1['kpi_rule']['avg_stock_value']:.0f}")
sh2 = shopmod.demo()
Yd, _ = sh2.matrix()
o_days = np.flatnonzero(rp1["plan"].ordered.sum(axis=0) > 0)
cday = int(o_days[o_days >= 30][0])                    # an order day: its orders are placed before its sales
cut = an.T - 8 * 7 + cday                              # from that replay day on, different sales
sh2.sales = sh2.sales[pd.to_datetime(sh2.sales["Date"]) < sh2.today - pd.Timedelta(days=an.T - cut)]
extra = pd.DataFrame([(sh2.today - pd.Timedelta(days=k), s_, 99) for k in range(1, an.T - cut + 1)
                      for s_ in sh2.products["SKU"]], columns=["Date", "SKU", "Qty"])
sh2.sales = pd.concat([sh2.sales, extra], ignore_index=True)
an2 = analysis.analyse(sh2)
rp2 = analysis.replay(an2, weeks=8)
ok("replay never looks ahead: different sales from an order day on leave every order up to it (its own too) "
   "unchanged", np.allclose(rp1["plan"].ordered[:, :cday + 1], rp2["plan"].ordered[:, :cday + 1]) and
   not np.allclose(rp1["plan"].ordered, rp2["plan"].ordered), f"order day {cday}")

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
