"""
Stock Planner
-------------
Shops, pharmacies and distributors run out of their best sellers while money sits in slow stock. Stock Planner
learns each product's sales pattern from the sales history, forecasts the coming weeks with a range, and says what to
order today from each supplier, within the budget, packs and minimum orders, to serve a chosen share of demand from
the shelf. A replay of the last weeks shows what it would have changed. Tested on the Walmart M5 data.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from stockplan import analysis, anim, report
from stockplan import shop as shopmod
from stockplan.uncertainty import quantiles

st.set_page_config(page_title="Stock Planner", page_icon="📦", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
C_PLAN, C_RULE, C_RED, C_OK, C_AMBER, C_VIOLET = "#2a78d6", "#9b9a95", "#e34948", "#1baf7a", "#eda100", "#6250d6"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

ss = st.session_state
for key, default in {"shop": None, "version": 0, "an": None, "an_version": -1, "rep": None, "rep_sig": None,
                     "target": 95, "keep_one": True, "use_budget": False, "budget": 100000, "low_days": 14,
                     "high_days": 28, "weeks": 13, "pick": None}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def layout(fig, height=300, **kw):
    top = 56 if kw.get("title") else 16
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=top, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def money(x, cur=None):
    cur = cur or (ss.shop.currency if ss.shop is not None else "")
    return f"{cur} {x:,.0f}"


def money_delta(x, cur=None):
    """'-Rs 431,525' / '+Rs 115,754' (the sign first, so the arrow points the right way)."""
    cur = cur or (ss.shop.currency if ss.shop is not None else "")
    return f"{'-' if x < 0 else '+'}{cur} {abs(x):,.0f}"


def kmoney(x):
    """A card value in thousands: (value, suffix, decimals)."""
    return (x / 1000, "k", 0) if x >= 10000 else ((x / 1000, "k", 1) if x >= 1000 else (x, "", 0))


def products(k):
    return f"{k} product" + ("" if k == 1 else "s")


def load(shop):
    ss.shop, ss.an, ss.rep, ss.rep_sig, ss.pick = shop, None, None, None, None
    ss.version = uuid.uuid4().hex     # cached results are keyed by this: unique to each load, never shared


def get_analysis():
    """Forecasts for the loaded shop (made once per set of sales data)."""
    if ss.shop is None:
        return None
    if ss.an is None or ss.an_version != ss.version:
        probs = shopmod.blocking(shopmod.check(ss.shop))
        if probs:
            return None
        bar = st.progress(0.0, "Reading the sales history…")
        ss.an = analysis.analyse(ss.shop, progress=lambda f, m: bar.progress(min(f, 1.0), m))
        ss.an_version = ss.version
        bar.empty()
    ss.an.shop = ss.shop
    return ss.an


def settings():
    return ss.target / 100, ss.keep_one, (float(ss.budget) if ss.use_budget else None)


@st.cache_data(show_spinner="Setting the order rule for your target…", max_entries=16)
def _shadow(_a, version, target, keep_one):
    return analysis.shadow(_a, target, keep_one)


@st.cache_data(show_spinner=False, max_entries=8)
def _orders(_a, key, target, keep_one, budget, lam):
    return analysis.order_today(_a, target, budget, keep_one, lam)


def orders(a):
    t, keep_one, budget = settings()
    lam = _shadow(a, ss.version, t, keep_one)
    stock = tuple(ss.shop.products["On hand"].astype(float)) + tuple(ss.shop.products["On order"].astype(float))
    return _orders(a, (ss.version, hash(stock)), t, keep_one, budget, lam)


@st.cache_data(show_spinner="Simulating stock for each service level…", max_entries=8)
def _frontier(_a, version, keep_one):
    return analysis.frontier(_a, keep_one=keep_one)


def replay_height(n):
    """Height of the shop replay: two grids of n squares side by side (as anim.shelf_replay lays them out)."""
    cols = int(np.ceil(np.sqrt(n * 2.2)))
    return int(215 + np.ceil(n / cols) * (514 // cols))


def html(doc, height):
    """An animated piece (our own HTML and JavaScript; shop data inside it is escaped)."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:
        components.html(doc, height=height, scrolling=False)


def product_label(a, i):
    p = a.shop.products.iloc[i]
    return f"{p['SKU']} · {p['Product']}"


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📦 Stock Planner")
    st.caption("What to order today, so the shelf serves your customers without tying up money in stock.")
    st.divider()
    side_steps = st.container()
    st.divider()
    st.markdown("**Ordering goal**")
    st.slider("Serve this share of demand from the shelf (%)", 80, 99, key="target",
              help="For the whole shop, in sales value: the share of what customers want that is on the shelf "
                   "when they ask. Stock goes where it sells most, so the shop reaches it with the least money in "
                   "stock. Higher means more stock (see ⚖️ Trade-off).")
    st.toggle("Keep at least one of every product sold in the last 3 months", key="keep_one",
              help="Customers find every recent product on the shelf, even very slow ones.")
    st.toggle("Limit today's spending", key="use_budget")
    if ss.use_budget:
        st.number_input("Budget for today's orders", min_value=0, step=5000, key="budget")
    st.divider()
    st.caption("Forecasts: LightGBM, one model for all products. Orders: every unit that is worth the money it "
               "ties up, lost sales. See **ℹ️ How it works**.")

st.title("📦 Stock Planner")
st.caption("Fewer empty shelves, less money in slow stock. From your sales history: a forecast for every product "
           "with a range, **what to order today** from each supplier (packs, minimum orders, budget), and a "
           "**replay of the last weeks** comparing it with the way you order now.")
tabs = st.tabs(["📥 Data", "🏪 Stock health", "🔮 Forecast", "🛒 Order today", "🎬 Replay", "⚖️ Trade-off",
                "🧪 Benchmark", "ℹ️ How it works"])

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demo: a pharmacy**")
        st.caption("83 products, 4 suppliers (lead times 2 to 14 days), two years of daily sales, holidays and "
                   "monsoon seasons. Fictional shop and numbers.")
        if st.button("Load the demo pharmacy", type="primary", width="stretch"):
            load(shopmod.demo())
            st.rerun()
    with c2:
        st.markdown("**Your shop**")
        st.caption("One Excel workbook: Products, Suppliers, Sales (date, SKU, quantity) and optional Events. "
                   "Point-of-sale exports can be pasted into the Sales sheet.")
        up = st.file_uploader("Workbook", type=["xlsx"], label_visibility="collapsed")
        if up is not None and st.button(f"Use {up.name}", type="primary", width="stretch"):
            try:
                load(shopmod.from_excel(up.getvalue()))
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read {up.name}: {exc}")
    with c3:
        st.markdown("**Template**")
        st.caption("The demo pharmacy as a workbook: keep the column names, replace the rows with your own.")
        st.download_button("Download the template (Excel)", lambda: shopmod.to_excel(shopmod.demo()),
                           "stock_template.xlsx", XLSX, width="stretch")
    shop = ss.shop
    if shop is None:
        st.info("Load the demo pharmacy or your own workbook to begin.")
    else:
        st.divider()
        st.subheader(shop.name)
        probs = shopmod.check(shop)
        block = shopmod.blocking(probs)
        Y, start = shop.matrix() if not block else (np.zeros((0, 0)), shop.today)
        m = st.columns(5)
        m[0].metric("Products", len(shop.products))
        m[1].metric("Suppliers", len(shop.suppliers))
        m[2].metric("Days of sales history", f"{Y.shape[1]:,}" if Y.size else "–")
        m[3].metric("Units sold", f"{Y.sum():,.0f}" if Y.size else "–")
        m[4].metric("Today", f"{shop.today.day} {shop.today:%b}", help=f"{shop.today:%A %d %B %Y}")
        for pr in probs:
            (st.error if pr in block else st.warning)(pr)
        if not block:
            st.success("✅ The data is complete: every product has a supplier, every number can be read.")
            tot = pd.Series(Y.sum(axis=0), index=pd.date_range(start, periods=Y.shape[1]))
            last = tot.iloc[-364:] if len(tot) >= 364 else tot
            wk = (last.index - last.index[0]).days // 7
            z = np.full((7, wk.max() + 1), np.nan)
            z[last.index.weekday, wk] = last.to_numpy()
            fig = go.Figure(go.Heatmap(z=z, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], xgap=2, ygap=2,
                                       colorscale=[[0, "rgba(42,120,214,.08)"], [1, C_PLAN]], showscale=False,
                                       customdata=np.array([[str((last.index[0] + pd.Timedelta(days=7 * w + d -
                                                                    last.index[0].weekday())).date())
                                                             for w in range(z.shape[1])] for d in range(7)]),
                                       hovertemplate="%{customdata}: %{z:,.0f} units<extra></extra>"))
            fig.update_xaxes(tickvals=[w for w in range(0, z.shape[1], 4)],
                             ticktext=[(last.index[0] + pd.Timedelta(days=7 * w)).strftime("%b %d")
                                       for w in range(0, z.shape[1], 4)])
            fig.update_yaxes(autorange="reversed")
            layout(fig, 230, title="Units sold each day, last 12 months")
            st.plotly_chart(fig, width="stretch")
            st.markdown("**Stock counts today** · edit them before planning if the shelf count changed")
            edit = st.data_editor(shop.products[["SKU", "Product", "On hand", "On order"]], hide_index=True,
                                  width="stretch", height=260, key=f"stock_{ss.version}",
                                  disabled=["SKU", "Product"],
                                  column_config={"On hand": st.column_config.NumberColumn(min_value=0, step=1),
                                                 "On order": st.column_config.NumberColumn(min_value=0, step=1)})
            new_oh = pd.to_numeric(edit["On hand"], errors="coerce").fillna(0).to_numpy(float)
            new_oo = pd.to_numeric(edit["On order"], errors="coerce").fillna(0).to_numpy(float)
            if not (np.allclose(new_oh, shop.products["On hand"].to_numpy(float)) and
                    np.allclose(new_oo, shop.products["On order"].to_numpy(float))):
                p = shop.products.copy()
                p["On hand"] = pd.to_numeric(edit["On hand"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
                p["On order"] = pd.to_numeric(edit["On order"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
                shop.products = p
                ss.rep = None
        e1, e2, e3 = st.columns(3)
        with e1.expander(f"Products ({len(shop.products)})"):
            st.dataframe(shop.products, hide_index=True, width="stretch")
        with e2.expander(f"Suppliers ({len(shop.suppliers)})"):
            st.dataframe(shop.suppliers, hide_index=True, width="stretch")
        with e3.expander(f"Events ({len(shop.events)})"):
            st.dataframe(shop.events, hide_index=True, width="stretch")
        st.download_button("Save this shop (Excel)", lambda: shopmod.to_excel(shop), "my_shop.xlsx", XLSX)

# everything below needs the forecasts
a = None
with tabs[1]:
    if ss.shop is None:
        st.info("Load a shop first (📥 Data).")
    else:
        a = get_analysis()
        if a is None:
            st.warning("Fix the data problems first (📥 Data).")
        else:
            o = orders(a)
            cls = analysis.classify(a.shop, a.Y)
            p = a.shop.products.reset_index(drop=True)
            stock_val = float(cls["stock_value"].sum())
            daily_cost = float((cls["avg_per_day"] * p["Unit cost"]).sum())
            risk = o["risk_before_arrival"].to_numpy(float) >= 0.2
            over = cls["days_of_cover"].to_numpy(float) > 90
            dead = cls["last_sale_days"].to_numpy() >= 60
            cur = a.shop.currency + " "
            v_over = float(cls.loc[over & ~dead, "stock_value"].sum())
            v_dead = float(cls.loc[dead, "stock_value"].sum())
            html(anim.kpi_cards([
                ("Stock value (at cost)", kmoney(stock_val)[0], cur, kmoney(stock_val)[1], kmoney(stock_val)[2],
                 products(len(p)), "accent"),
                ("Days of stock", stock_val / max(daily_cost, 1e-9), "", " days", 0, "at the last 4 weeks' sales",
                 "accent"),
                ("May run out before the next delivery", int(risk.sum()), "", " products", 0,
                 "20% chance or more", "out"),
                ("Too much stock (over 3 months)", kmoney(v_over)[0], cur, kmoney(v_over)[1], kmoney(v_over)[2],
                 products(int((over & ~dead).sum())), "over"),
                ("No sale for 2 months", kmoney(v_dead)[0], cur, kmoney(v_dead)[1], kmoney(v_dead)[2],
                 products(int(dead.sum())), "low")], DARK), 150)
            g1, g2 = st.columns([3, 2])
            with g1:
                cov = np.clip(np.nan_to_num(cls["days_of_cover"].to_numpy(float), posinf=365), 0.5, 365)
                tm = pd.DataFrame({"Category": p["Category"], "Product": p["Product"],
                                   "value": np.maximum(cls["stock_value"].to_numpy(float), 1.0),
                                   "cover": np.log10(cov), "days": cov.round(0)})
                fig = go.Figure(go.Treemap(
                    labels=list(tm["Category"].unique()) + list(tm["Product"]),
                    parents=[""] * tm["Category"].nunique() + list(tm["Category"]),
                    values=[0] * tm["Category"].nunique() + list(tm["value"]),
                    marker=dict(colors=[None] * tm["Category"].nunique() + list(tm["cover"]),
                                colorscale=[[0, C_RED], [0.35, C_AMBER], [0.55, C_OK], [0.8, C_OK], [1, C_VIOLET]],
                                cmin=np.log10(0.5), cmax=np.log10(365), showscale=False,
                                line=dict(width=1, color="rgba(128,128,128,.4)")),
                    customdata=[0] * tm["Category"].nunique() + list(tm["days"]),
                    hovertemplate="<b>%{label}</b><br>stock value %{value:,.0f}<br>%{customdata} days of stock"
                                  "<extra></extra>", branchvalues="remainder", textinfo="label"))
                layout(fig, 430, title="Where the money sits: stock value by product, coloured by days of stock")
                st.plotly_chart(fig, width="stretch")
                st.caption("Red: a few days left · green: 1 to 3 months · violet: more than 6 months.")
            with g2:
                abc, xyz = ["A", "B", "C"], ["X", "Y", "Z"]
                z = np.array([[((cls["ABC"] == r) & (cls["XYZ"] == c)).sum() for c in xyz] for r in abc])
                v = np.array([[cls.loc[(cls["ABC"] == r) & (cls["XYZ"] == c), "sales_value_90d"].sum() for c in xyz]
                              for r in abc])
                share = v / max(v.sum(), 1e-9)
                fig = go.Figure(go.Heatmap(z=share, x=["X steady", "Y variable", "Z erratic"],
                                           y=["A top 80% of sales", "B next 15%", "C last 5%"],
                                           text=[[f"{z[i, j]} products<br>{share[i, j]:.0%} of sales" for j in range(3)]
                                                 for i in range(3)], texttemplate="%{text}", showscale=False,
                                           colorscale=[[0, "rgba(42,120,214,.06)"], [1, C_PLAN]], xgap=3, ygap=3,
                                           hoverinfo="skip"))
                fig.update_yaxes(autorange="reversed")
                layout(fig, 300, title="Products by sales value and by how steady they sell")
                st.plotly_chart(fig, width="stretch")
                st.caption("A and X products sell most and most predictably; Z products need more safety stock "
                           "for the same service.")
            ht = report.health_table(a.shop, cls, o)
            st.markdown("**Needs attention**")
            attn = ht[ht["Status"] != "fine"].sort_values(["Status", "Stock value"], ascending=[True, False])
            st.dataframe(attn, hide_index=True, width="stretch", height=300, column_config={
                "Risk of running out": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Stock value": st.column_config.NumberColumn(format="%,.0f")})

# ---------------------------------------------------------------------------
# 3. Forecast
# ---------------------------------------------------------------------------
with tabs[2]:
    a = ss.an if ss.shop is not None and ss.an is not None and ss.an_version == ss.version else None
    if a is None:
        st.info("Load a shop first (📥 Data).")
    else:
        o = orders(a)
        acc = a.accuracy
        e_eng = acc["error_engine"].sum() / max(acc["sold"].sum(), 1e-9)
        e_avg = acc["error_average"].sum() / max(acc["sold"].sum(), 1e-9)
        top = int(np.argmax(a.classes["sales_value_90d"].to_numpy()))
        opts = list(range(a.n))
        if ss.pick is None or ss.pick >= a.n:
            ss.pick = top
        i = st.selectbox("Product", opts, format_func=lambda k: product_label(a, k), key="pick")
        p = a.shop.products.iloc[i]
        m = st.columns(4)
        m[0].metric("Weekly forecast error, all products", f"{e_eng:.0%}",
                    delta=f"{(e_eng - e_avg) * 100:+.0f} points vs 4-week average",
                    delta_color="inverse", help=f"On the last {a.H // 7} weeks, which the model had not seen: "
                                                "sum of |actual - forecast| per product and week, over units sold.")
        e_i = acc["error_engine"].iloc[i] / max(acc["sold"].iloc[i], 1e-9)
        m[1].metric("This product", f"{e_i:.0%}" if acc["sold"].iloc[i] > 0 else "–",
                    help="Same measure for this product alone (high for slow sellers: a few units either way).")
        m[2].metric(f"Forecast, next {a.H} days", f"{a.mu[i].sum():,.0f} units",
                    help="Mean of the forecast; the band in the chart shows the likely range.")
        m[3].metric("On hand / on order", f"{p['On hand']:,.0f} / {p['On order']:,.0f}")
        g1, g2 = st.columns([3, 2])
        with g1:
            T, H = a.T, a.H
            show = min(112, T)
            dates_h = pd.date_range(a.start + pd.Timedelta(days=T - show), periods=show)
            dates_f = pd.date_range(a.shop.today, periods=H)
            per = a.mu[i].mean()
            qd = quantiles(a.mu[i], a.disp.phi(np.full(H, per), 1), [0.05, 0.25, 0.75, 0.95])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates_h, y=a.Y[i, T - show:], name="sold", marker=dict(color=C_RULE),
                                 hovertemplate="%{x|%a %d %b}: %{y:.0f} sold<extra></extra>"))
            fig.add_trace(go.Scatter(x=dates_h[-H:], y=a.holdout_mean[i], name="forecast made without these weeks",
                                     line=dict(color=C_PLAN, width=2, dash="dot"),
                                     hovertemplate="%{x|%a %d %b}: %{y:.1f} forecast<extra></extra>"))
            fig.add_trace(go.Scatter(x=list(dates_f) + list(dates_f[::-1]), y=list(qd[0.95]) + list(qd[0.05][::-1]),
                                     fill="toself", fillcolor="rgba(42,120,214,.12)", line=dict(width=0),
                                     name="90% range", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=list(dates_f) + list(dates_f[::-1]), y=list(qd[0.75]) + list(qd[0.25][::-1]),
                                     fill="toself", fillcolor="rgba(42,120,214,.25)", line=dict(width=0),
                                     name="50% range", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dates_f, y=a.mu[i], name="forecast", line=dict(color=C_PLAN, width=2.5),
                                     hovertemplate="%{x|%a %d %b}: %{y:.1f} expected<extra></extra>"))
            ev = a.shop.events.copy()
            if len(ev):
                ev["Date"] = pd.to_datetime(ev["Date"])
                for _, r in ev[(ev["Date"] >= dates_h[0]) & (ev["Date"] <= dates_f[-1])].iterrows():
                    fig.add_vline(x=r["Date"], line=dict(color="rgba(128,128,128,.45)", dash="dot", width=1))
            fig.add_vline(x=a.shop.today, line=dict(color=C_PLAN, width=1))
            layout(fig, 380, title=f"{p['Product']}: units a day", barmode="overlay")
            st.plotly_chart(fig, width="stretch")
            st.caption("Grey bars: sales. Dotted: what the model forecast for the last weeks before it saw them. "
                       "After today: the forecast with its likely range, moved part of the way by how far the last "
                       f"weeks' sales were from it (×{a.level[i]:.2f} for this product). Dotted lines: holidays and "
                       "events.")
        with g2:
            row = o.iloc[i]
            lam = _shadow(a, ss.version, *settings()[:2])
            q = float(row["qty"]) if row["today"] else float(analysis.what_if_today(a, i, lam, ss.keep_one))
            f0 = analysis.futures(a, i, 0.0, size=60, seed=1)
            f1 = analysis.futures(a, i, q, size=60, seed=1)
            st.markdown(f"**Possible futures of this shelf** · until the next delivery after today's order "
                        f"({int(a.lead[i])} days lead time, ordered every {int(a.review[i])} days)")
            html(anim.futures(f0["stock"], f1["stock"], a.lead[i], a.review[i], a.shop.today.strftime("%d %b"),
                              int(q), dark=DARK), 380)
            st.caption("60 futures drawn around the forecast with the measured range. Red: the shelf runs empty. "
                       + ("Today's order: " if row["today"] else "An order today (its supplier's next order day is "
                          f"{row['note'][14:]}): ") + f"{int(q)} units.")
        wd = pd.DataFrame({"day": pd.date_range(a.start, periods=a.T).weekday, "y": a.Y[i]}).iloc[-182:]
        prof = wd.groupby("day")["y"].mean().reindex(range(7), fill_value=0)
        fig = go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=prof.to_numpy(),
                               marker=dict(color=C_PLAN, cornerradius=4),
                               hovertemplate="%{x}: %{y:.1f} a day<extra></extra>"))
        layout(fig, 220, title="Average sales by weekday, last 6 months")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 4. Order today
# ---------------------------------------------------------------------------
with tabs[3]:
    a = ss.an if ss.shop is not None and ss.an is not None and ss.an_version == ss.version else None
    if a is None:
        st.info("Load a shop first (📥 Data).")
    else:
        o = orders(a)
        t, keep_one, budget = settings()
        sup = a.shop.suppliers.set_index("Supplier")
        todays = sorted(o.loc[o["today"], "supplier"].unique())
        st.markdown(f"**{a.shop.today.strftime('%A %d %B %Y')}** · order day for: "
                    + (", ".join(todays) if todays else "no supplier"))
        if todays:
            cols = st.columns(len(todays))
            for k, s in enumerate(todays):
                g = o[(o["supplier"] == s) & o["today"]]
                val = float(g["value"].sum())
                mn = float(sup.loc[s, "Min order value"])
                with cols[k]:
                    st.metric(s, money(val), delta=f"{int((g['qty'] > 0).sum())} products · arrives in "
                                                   f"{int(sup.loc[s, 'Lead time (days)'])} days", delta_color="off")
                    if val == 0:
                        st.caption("Nothing needed today." if not g["note"].str.contains("waits").any() else
                                   "Order waits: below the supplier's minimum and nothing is urgent.")
                    elif val >= mn:
                        st.caption(f"✓ above the minimum order ({money(mn)})")
                    else:
                        st.caption(f"⚠ below the minimum order ({money(mn)})")
            tot = float(o.loc[o["today"], "value"].sum())
            if budget is not None:
                st.progress(min(tot / max(budget, 1), 1.0), f"Spending {money(tot)} of the budget {money(budget)}")
            lines = o[o["today"] & (o["qty"] > 0)]
            if len(lines):
                g1, g2 = st.columns([2, 3])
                with g1:
                    sups = sorted(lines["supplier"].unique())
                    cats = sorted(lines["Category"].unique())
                    lab = ["Today's orders"] + sups + cats
                    src, dst, val = [], [], []
                    for s_ in sups:
                        src.append(0)
                        dst.append(1 + sups.index(s_))
                        val.append(float(lines.loc[lines["supplier"] == s_, "value"].sum()))
                    for (s_, c_), g in lines.groupby(["supplier", "Category"]):
                        src.append(1 + sups.index(s_))
                        dst.append(1 + len(sups) + cats.index(c_))
                        val.append(float(g["value"].sum()))
                    fig = go.Figure(go.Sankey(node=dict(label=lab, pad=14, thickness=14,
                                                        color=[C_PLAN] + [C_VIOLET] * len(sups) + [C_OK] * len(cats)),
                                              link=dict(source=src, target=dst, value=val,
                                                        color="rgba(42,120,214,.25)"),
                                              valueformat=",.0f"))
                    layout(fig, 360, title=f"Where today's {money(tot)} goes")
                    st.plotly_chart(fig, width="stretch")
                with g2:
                    show = pd.DataFrame({
                        "Product": lines["Product"], "Supplier": lines["supplier"], "On hand": lines["on_hand"],
                        "Expected sales until the next delivery": lines["m_lr"].round(0),
                        "Order": lines["qty"], "Packs": (lines["qty"] / lines["pack"]).round(1),
                        "Value": lines["value"].round(0), "Served without": lines["fill_without"].astype(float),
                        "Served with": lines["fill_with"].astype(float),
                        "Last unit sells": lines["last_unit_chance"].astype(float), "Note": lines["note"]})
                    st.dataframe(show, hide_index=True, width="stretch", height=360, column_config={
                        "Served without": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                        "Served with": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                        "Last unit sells": st.column_config.NumberColumn(format="percent",
                                                                         help="Chance that the last unit ordered "
                                                                              "sells before the next delivery"),
                        "Value": st.column_config.NumberColumn(format="%,.0f")})
                    st.caption(f"Each unit is ordered while its chance of selling before the next delivery is worth "
                               f"the money it ties up; one rule for the whole shop, set so that {t:.0%} of the "
                               "demand value is served. **Served**: this product's share of demand met until the "
                               "next delivery, without and with the order. Whole packs, at least the minimum order "
                               "quantity.")
                st.download_button("📄 Purchase orders (Excel, one sheet per supplier)",
                                   lambda o=o, s=a.shop: report.purchase_orders(o, s),
                                   f"orders_{a.shop.today.date()}.xlsx", XLSX, type="primary")
            else:
                st.success("Nothing needs ordering today: the stock covers the target until the next delivery.")
        watch = o[~o["today"] & (o["risk_before_arrival"] >= 0.1)].sort_values("risk_before_arrival", ascending=False)
        st.markdown("**Watch list** · other suppliers' products that may run out before their next delivery")
        if len(watch):
            st.dataframe(pd.DataFrame({"Product": watch["Product"], "Supplier": watch["supplier"],
                                       "On hand": watch["on_hand"], "Risk of running out": watch["risk_before_arrival"]
                                       .astype(float), "When": watch["note"]}), hide_index=True, width="stretch",
                         column_config={"Risk of running out": st.column_config.ProgressColumn(
                             format="percent", min_value=0, max_value=1)})
            st.caption("Consider an extra order or asking the supplier for an earlier delivery.")
        else:
            st.caption("None: every other product is likely to last until its next delivery.")

# ---------------------------------------------------------------------------
# 5. Replay
# ---------------------------------------------------------------------------
with tabs[4]:
    a = ss.an if ss.shop is not None and ss.an is not None and ss.an_version == ss.version else None
    if a is None:
        st.info("Load a shop first (📥 Data).")
    else:
        st.markdown("**The last weeks again, ordered two ways.** Real sales, real lead times, packs and minimum "
                    "quantities: your way of ordering against Stock Planner (its model learnt only from the sales "
                    "before these weeks).")
        r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
        r1.slider("Your way: reorder when stock covers fewer than … days", 3, 42, key="low_days")
        r2.slider("… and then order up to … days of sales", 7, 90, key="high_days")
        r3.select_slider("Weeks to replay", [4, 8, 13, 26], key="weeks")
        go_btn = r4.button("▶ Replay", type="primary", width="stretch")
        sig = (ss.version, ss.low_days, ss.high_days, ss.weeks, ss.target, ss.keep_one)
        if go_btn:
            bar = st.progress(0.0, "Replaying…")
            try:
                ss.rep = analysis.replay(a, ss.low_days, max(ss.high_days, ss.low_days + 1), ss.target / 100,
                                         ss.weeks, ss.keep_one, progress=lambda f, m_: bar.progress(min(f, 1.0), m_))
                ss.rep_sig = sig
            except ValueError as exc:
                st.error(str(exc))
            bar.empty()
        r = ss.rep
        if r is None:
            st.info("Set your usual way of ordering and press **Replay**.")
        else:
            if ss.rep_sig != sig:
                st.warning("The settings changed since this replay: press Replay again.")
            kp, kr = r["kpi_plan"], r["kpi_rule"]
            gm_p = float((r["plan"].sold.sum(axis=1) * (r["price"] - r["cost"])).sum())
            gm_r = float((r["rule"].sold.sum(axis=1) * (r["price"] - r["cost"])).sum())
            m = st.columns(4)
            m[0].metric("Demand served from the shelf", f"{kp['value_fill']:.1%}",
                        delta=f"{(kp['value_fill'] - kr['value_fill']) * 100:+.1f} points vs your way "
                              f"({kr['value_fill']:.1%})", help="Share of the demand, in sales value, met from the shelf.")
            m[1].metric("Sales lost to empty shelves", money(kp["lost_sales_value"]),
                        delta=f"{money_delta(kp['lost_sales_value'] - kr['lost_sales_value'])} vs your way",
                        delta_color="inverse")
            m[2].metric("Average stock value", money(kp["avg_stock_value"]),
                        delta=f"{(kp['avg_stock_value'] / max(kr['avg_stock_value'], 1) - 1) * 100:+.0f}% vs your way",
                        delta_color="inverse")
            m[3].metric("Gross margin earned", money(gm_p), delta=f"{money_delta(gm_p - gm_r)} vs your way")
            daily = np.maximum(a.Y[:, a.T - r["days"] - 28:a.T - r["days"]].mean(axis=1), 1e-3)
            dates = [(r["start"] + pd.Timedelta(days=d)).strftime("%a %d %b") for d in range(r["days"])]
            html(anim.shelf_replay(a.shop.products["Product"].tolist(),
                                   anim.status_codes(r["rule"].on_hand, daily), anim.status_codes(r["plan"].on_hand, daily),
                                   (r["rule"].lost * r["price"][:, None]).sum(axis=0),
                                   (r["plan"].lost * r["price"][:, None]).sum(axis=0),
                                   (r["rule"].on_hand * r["cost"][:, None]).sum(axis=0),
                                   (r["plan"].on_hand * r["cost"][:, None]).sum(axis=0), dates,
                                   (f"Your way ({ss.low_days} → {ss.high_days} days)", "Stock Planner"),
                                   a.shop.currency, DARK), replay_height(a.n))
            st.caption(f"Replayed {r['days']} days from {r['start']:%d %b %Y}. Both start from a full shelf for "
                       "their own way of ordering; the planner's forecasts for each order day use only sales before "
                       "that day.")

# ---------------------------------------------------------------------------
# 6. Trade-off
# ---------------------------------------------------------------------------
with tabs[5]:
    a = ss.an if ss.shop is not None and ss.an is not None and ss.an_version == ss.version else None
    if a is None:
        st.info("Load a shop first (📥 Data).")
    else:
        fr = _frontier(a, ss.version, ss.keep_one)
        fr = fr[(fr["target"] >= 0.75) & (fr["target"] <= 0.998)].reset_index(drop=True)
        cur = ss.target / 100
        y_cur = float(np.interp(cur, fr["target"], fr["stock_value"]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fr["target"] * 100, y=fr["stock_value"], mode="lines+markers",
                                 line=dict(color=C_PLAN, width=3, shape="spline"), marker=dict(size=7),
                                 name="Stock Planner", hovertemplate="%{x:.1f}% served: %{y:,.0f} stock<extra></extra>"))
        fig.add_trace(go.Scatter(x=[cur * 100], y=[y_cur], mode="markers", name="your target",
                                 marker=dict(size=16, color=C_PLAN, line=dict(color="white", width=2))))
        if ss.rep is not None:
            kr, kp = ss.rep["kpi_rule"], ss.rep["kpi_plan"]
            fig.add_trace(go.Scatter(x=[kr["value_fill"] * 100], y=[kr["avg_stock_value"]], mode="markers+text",
                                     marker=dict(size=14, color=C_RULE, symbol="diamond"), text=["your way (replay)"],
                                     textposition="top center", name="your way, replayed"))
            fig.add_trace(go.Scatter(x=[kp["value_fill"] * 100], y=[kp["avg_stock_value"]], mode="markers+text",
                                     marker=dict(size=14, color=C_PLAN, symbol="diamond-open"),
                                     text=["Stock Planner (replay)"], textposition="bottom center",
                                     name="Stock Planner, replayed"))
        frames = [go.Frame(data=[go.Scatter(x=fr["target"][:k + 1] * 100, y=fr["stock_value"][:k + 1])],
                           traces=[0], name=str(k)) for k in range(len(fr))]
        fig.frames = frames
        fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=1, y=1.2, xanchor="right",
                                            buttons=[dict(label="▶ Draw the curve", method="animate",
                                                          args=[None, dict(frame=dict(duration=260, redraw=False),
                                                                           transition=dict(duration=200),
                                                                           fromcurrent=False)])])])
        layout(fig, 420, title="How much stock each service level needs (average stock value)")
        fig.update_xaxes(title="share of demand served from the shelf (%)")
        fig.update_yaxes(title=f"average stock value ({a.shop.currency})", rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
        v95, v98 = (float(np.interp(x, fr["target"], fr["stock_value"])) for x in (0.95, 0.98))
        v90 = float(np.interp(0.90, fr["target"], fr["stock_value"]))
        m = st.columns(3)
        m[0].metric("At your target", money(y_cur), help="Average stock value needed (steady state).")
        m[1].metric("From 90% to 95% served", f"+{money(v95 - v90)}", delta=f"+{(v95 / v90 - 1) * 100:.0f}% stock",
                    delta_color="off")
        m[2].metric("From 95% to 98% served", f"+{money(v98 - v95)}", delta=f"+{(v98 / v95 - 1) * 100:.0f}% stock",
                    delta_color="off")
        st.caption("Each extra point of service costs more stock than the last: the curve bends upwards. From "
                   "simulating the coming weeks over possible futures: share of the demand value served, and the "
                   "average stock value. Diamonds: the replay of the last weeks (actual sales).")

# ---------------------------------------------------------------------------
# 7. Benchmark
# ---------------------------------------------------------------------------
with tabs[6]:
    path = os.path.join(HERE, "benchmarks", "results_m5.json")
    if not os.path.exists(path):
        st.info("No benchmark results yet: run `python benchmarks/run_m5.py forecast` then `score`.")
    else:
        R = json.load(open(path))
        acc = R["accuracy"]
        st.markdown("#### Walmart M5: 30,490 product-store series, 28 days ahead")
        st.caption("The M5 competition (2020, 5,558 teams): forecast 28 days of unit sales for 3,049 products in 10 "
                   "stores. Scored with the official measure (WRMSSE over 12 levels, from the total down to each "
                   "product in each store) and the official weights. The number of trees was chosen on the earlier "
                   "public-leaderboard period; the final test was scored once.")
        m = st.columns(4)
        m[0].metric("Stock Planner (WRMSSE, lower is better)", f"{acc['engine']:.3f}")
        m[1].metric("Best official benchmark (ES_bu)", f"{acc['benchmarks']['ES_bu']:.3f}",
                    delta=f"{(acc['engine'] / acc['benchmarks']['ES_bu'] - 1) * 100:+.1f}%", delta_color="inverse")
        m[2].metric("Seasonal naive", f"{acc['benchmarks']['sNaive']:.3f}")
        m[3].metric("Winner (YJ_STU)", f"{acc['top'][0][1]:.3f}", delta=acc.get("rank_text", ""), delta_color="off")
        names = ["Naive", "sNaive", "ARIMA_bu", "ES_bu"] + ["50th place", "10th place", "1st place"] + ["Stock Planner"]
        vals = [acc["benchmarks"][k] for k in ("Naive", "sNaive", "ARIMA_bu", "ES_bu")] + \
            [acc["top"][49][1], acc["top"][9][1], acc["top"][0][1], acc["engine"]]
        cols = [C_RULE] * 4 + ["rgba(98,80,214,.55)"] * 3 + [C_PLAN]
        fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker=dict(color=cols, cornerradius=4),
                               text=[f"{v:.3f}" for v in vals], textposition="outside",
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        layout(fig, 340, title="WRMSSE on the final test period (lower is better)")
        fig.update_xaxes(range=[0, max(vals) * 1.15])
        st.plotly_chart(fig, width="stretch")
        inv = R.get("inventory")
        if inv:
            st.markdown("#### Ordering with the forecasts: the same 30,490 shelves, 28 days")
            st.caption(inv["setup"])
            comps = inv.get("same_fill", [])
            if comps:
                fig = go.Figure()
                xs = [f"{c[1]:.1%} served<br><sub>rule: {c[0]:g}× cover</sub>" for c in comps]
                fig.add_trace(go.Bar(x=xs, y=[c[2] for c in comps], name="rule of thumb",
                                     marker=dict(color=C_RULE, cornerradius=4),
                                     text=[f"{c[2]:.1f} d" for c in comps], textposition="outside",
                                     hovertemplate="rule of thumb: %{y:.1f} days of stock<extra></extra>"))
                wo = {round(c[1], 4): c for c in inv.get("same_fill_without", [])}
                if wo:
                    fig.add_trace(go.Bar(x=xs, y=[wo[round(c[1], 4)][3] if round(c[1], 4) in wo else None
                                                  for c in comps], name="Stock Planner without the latest weeks",
                                         marker=dict(color="rgba(42,120,214,.4)", cornerradius=4),
                                         hovertemplate="without the latest weeks: %{y:.1f} days<extra></extra>"))
                fig.add_trace(go.Bar(x=xs, y=[c[3] for c in comps], name="Stock Planner",
                                     marker=dict(color=C_PLAN, cornerradius=4),
                                     text=["same" if abs(c[4]) < 0.03 else f"{-c[4] * 100:+.0f}%" for c in comps],
                                     textposition="outside", textfont=dict(size=14),
                                     hovertemplate="Stock Planner: %{y:.1f} days of stock<extra></extra>"))
                layout(fig, 400, title="Stock needed for the same service as the rule of thumb (days of sales)",
                       barmode="group")
                fig.update_yaxes(title="average stock, days of sales", rangemode="tozero",
                                 range=[0, max(c[2] for c in comps) * 1.2])
                st.plotly_chart(fig, width="stretch")
            st.caption(inv["summary"])
            with st.expander("All the points: service against stock for each way of ordering"):
                fig = go.Figure()
                series = [("rule", "rule of thumb", C_RULE, "solid", 3, lambda r: f"{r['cover']:g}× cover"),
                          ("per_product", "same target for every product", C_AMBER, "dot", 2, None),
                          ("without_correction", "Stock Planner without the latest weeks", C_PLAN, "dash", 1.5, None),
                          ("planner", "Stock Planner", C_PLAN, "solid", 3, lambda r: f"target {r['target']:.0%}")]
                for key, name, colour, dash, width, lab in series:
                    if key not in inv:
                        continue
                    df = pd.DataFrame(inv[key]).sort_values("stock_days")
                    fig.add_trace(go.Scatter(
                        x=df["stock_days"], y=df["value_fill"] * 100, mode="lines+markers+text" if lab else
                        "lines+markers", line=dict(color=colour, width=width, dash=dash),
                        marker=dict(size=8 if width > 2 else 6), name=name,
                        text=[lab(r) for r in df.to_dict("records")] if lab else None,
                        textposition="top left" if key == "rule" else "bottom right",
                        hovertemplate=name + ": %{y:.1f}% served with %{x:.1f} days of stock<extra></extra>"))
                layout(fig, 430, title="Demand value served against stock held (up and to the left is better)")
                fig.update_xaxes(title="average stock, in days of sales")
                fig.update_yaxes(title="demand value served (%)")
                st.plotly_chart(fig, width="stretch")
            cal = pd.DataFrame(inv["planner"])
            lw = R.get("latest_weeks", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Target against what the final test gave**")
                st.dataframe(pd.DataFrame({"Target": [f"{t:.0%}" for t in cal["target"]],
                                           "Served": [f"{v:.1%}" for v in cal["value_fill"]],
                                           "Days of stock": [f"{v:.1f}" for v in cal["stock_days"]],
                                           "Shelves that ran out": [f"{v:.0%}" for v in cal["stockout_share"]]}),
                             hide_index=True, width="stretch")
                st.caption("Shelves that ran out: products that were empty at least once in the 28 days.")
            with c2:
                if lw:
                    st.markdown("**How much to trust the latest weeks** (chosen before the final test)")
                    tab = {"Correction": [f"{float(k):.0%}" + (" ✓" if float(k) == float(lw["chosen"]) else "")
                                          for k in lw["weights"]]}
                    for j, f in enumerate(lw["fills"]):
                        tab[f"{f:.0%} served"] = [f"{v['stock_days_at'][j]:.2f} d"
                                                  if v["stock_days_at"][j] is not None else "–"
                                                  for v in lw["weights"].values()]
                    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
                    st.caption(f"Days of stock needed on the earlier public leaderboard period. The forecast's "
                               f"inputs are at least 28 days old; the latest weeks' error moves it "
                               f"{float(lw['chosen']):.0%} of the way. {lw.get('summary', '')}")
        unc = R.get("uncertainty")
        if unc:
            st.markdown("#### Ranges: product-store level (level 12) of the M5 uncertainty competition")
            names = list(unc["benchmarks"]) + ["Stock Planner"]
            vals = list(unc["benchmarks"].values()) + [unc["engine"]]
            fig = go.Figure(go.Bar(x=names, y=vals, marker=dict(color=[C_RULE] * len(unc["benchmarks"]) + [C_PLAN],
                                                                 cornerradius=4),
                                   text=[f"{v:.3f}" for v in vals], textposition="outside"))
            layout(fig, 300, title="Scaled pinball loss over 9 quantiles, level 12 (lower is better)")
            fig.update_yaxes(range=[0, max(vals) * 1.2])
            st.plotly_chart(fig, width="stretch")
            st.caption(unc["note"])

# ---------------------------------------------------------------------------
# 8. How it works
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("""
#### The problem
A shop orders by feel or by a simple rule ("when it covers less than two weeks, order four weeks"). Best sellers run
out before the delivery arrives, slow products pile up, and nobody can say how much stock a given service needs.

#### 1. Forecast with a range
One gradient-boosted tree model (LightGBM) learns from all products at once: recent days and weeks, the same weekday
in past weeks, last year at this time, holidays and events, and the product's category and supplier. Every input is
at least as old as the forecast horizon, so one model forecasts all the days up to the next delivery at once and can
replay past weeks honestly. The latest weeks are brought in afterwards: the forecast moves a quarter of the way
towards how far the last 4 weeks' sales were from what it had forecast for them (a product picking up gets more). The
range around the forecast is a negative binomial distribution whose width is measured on recent weeks the model had
not seen, separately for 1, 7, 14 and 28 days.

#### 2. Order today
For each product whose supplier takes orders today: the stock on the shelf, what is on order, the demand until the
order lands (lead time) and until the next one lands (lead time + order interval). A customer who finds an empty shelf
is a lost sale, so the stock when the order lands is what is left after the lead time (never below zero) plus the
order. From that comes, for every extra unit, its **chance of selling before the next delivery**.

One rule for the whole shop: a unit is ordered while its price times its chance of selling is worth the money it
ties up (its cost times one shop-wide factor). The factor is set so that the shop serves your target share of the
demand value with the **least money in stock**: steady sellers get deep stock, slow and erratic ones only what is
likely to sell, so the target costs much less stock than giving every product the same service. Quantities are
whole packs and at least the minimum order quantity; optionally one of every product sold in the last 3 months stays
on the shelf.

**Budget:** if today's orders cost more than the budget, the packs that sell the most value per unit of money go
first. **Supplier minimums:** an order below a supplier's minimum is topped up with the most valuable extra packs if
something is likely to run out and the budget allows, otherwise it waits for the next order day.

#### 3. Replay
The model is trained again on the sales before the replayed weeks only; each order day then uses only what was known
on that day. Real sales are the demand; empty shelves lose sales. Your way of ordering runs on the same days.

#### Tested on Walmart's M5 data
See 🧪 Benchmark: the official accuracy measure on 30,490 series, and the ordering policy on the same shelves.
""")

# sidebar steps (filled last, after any work in this run)
with side_steps:
    done = [ss.shop is not None, ss.an is not None and ss.an_version == ss.version, ss.rep is not None]
    for ok, lab in zip(done, ["Data loaded", "Forecasts made", "Replay run"]):
        st.markdown(f"{'✅' if ok else '⬜'} {lab}")
    if ss.shop is not None:
        st.caption(ss.shop.name)
