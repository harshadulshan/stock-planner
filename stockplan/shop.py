"""
A shop's data: products, suppliers, daily sales and events, as tables and as an Excel workbook, plus the fictional
demo pharmacy.

Products  : SKU, Product, Category, Supplier, Unit cost, Price, On hand, On order, Pack size, Min order qty
Suppliers : Supplier, Lead time (days), Order every (days), Next order date, Min order value
Sales     : Date, SKU, Qty               (one row per product and day with sales; missing days count as 0)
Events    : Date, Event                  (optional: holidays, festivals, promotions)
Settings  : Shop, Currency, Today, Budget, Target fill rate
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRODUCT_COLS = ["SKU", "Product", "Category", "Supplier", "Unit cost", "Price", "On hand", "On order", "Pack size",
                "Min order qty"]
SUPPLIER_COLS = ["Supplier", "Lead time (days)", "Order every (days)", "Next order date", "Min order value"]


@dataclass
class Shop:
    name: str
    currency: str
    today: pd.Timestamp                      # the first day without sales data (decisions are made today)
    products: pd.DataFrame
    suppliers: pd.DataFrame
    sales: pd.DataFrame                      # Date, SKU, Qty
    events: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["Date", "Event"]))
    budget: float | None = None
    target: float = 0.95

    def matrix(self):
        """Units sold per product (rows, in Products order) and day, from the first sales date to the day before
        today: (Y, start)."""
        s = self.sales.copy()
        s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
        start = s["Date"].min()
        T = (self.today - start).days
        pos = pd.Series(np.arange(len(self.products)), index=self.products["SKU"].astype(str))
        s = s[s["SKU"].astype(str).isin(pos.index) & (s["Date"] < self.today)]
        Y = np.zeros((len(self.products), T), np.float32)
        np.add.at(Y, (pos[s["SKU"].astype(str)].to_numpy(), (s["Date"] - start).dt.days.to_numpy()),
                  s["Qty"].to_numpy(np.float32))
        return Y, start

    def supplier_table(self):
        sup = self.suppliers.set_index("Supplier")
        return sup


# ------------------------------------------------------------------------------------------------ checks
def check(shop: Shop) -> list:
    """Problems that stop planning (empty list = fine)."""
    out = []
    p, s = shop.products, shop.suppliers
    miss = [c for c in PRODUCT_COLS if c not in p.columns]
    if miss:
        return [f"Products: missing column(s) {', '.join(miss)}"]
    miss = [c for c in SUPPLIER_COLS if c not in s.columns]
    if miss:
        return [f"Suppliers: missing column(s) {', '.join(miss)}"]
    if p["SKU"].astype(str).duplicated().any():
        out.append("Products: the same SKU appears twice: " +
                   ", ".join(p.loc[p["SKU"].astype(str).duplicated(), "SKU"].astype(str).head(3)))
    unknown = sorted(set(p["Supplier"].astype(str)) - set(s["Supplier"].astype(str)))
    if unknown:
        out.append(f"Products: supplier(s) not in Suppliers: {', '.join(unknown[:3])}")
    for c in ("Unit cost", "Price", "On hand", "On order", "Pack size", "Min order qty"):
        v = pd.to_numeric(p[c], errors="coerce")
        if v.isna().any() or (v < 0).any():
            out.append(f"Products: '{c}' must be a number of 0 or more (row {int(np.flatnonzero(v.isna() | (v < 0))[0]) + 2})")
    if (pd.to_numeric(p["Pack size"], errors="coerce") < 1).any():
        out.append("Products: 'Pack size' must be at least 1")
    for c in ("Lead time (days)", "Order every (days)"):
        v = pd.to_numeric(s[c], errors="coerce")
        if v.isna().any() or (v < 0).any() or (c == "Order every (days)" and (v < 1).any()):
            out.append(f"Suppliers: '{c}' must be a whole number of days (at least {1 if 'every' in c else 0})")
    if len(shop.sales) == 0:
        out.append("Sales: no sales rows")
    else:
        d = pd.to_datetime(shop.sales["Date"], errors="coerce")
        if d.isna().any():
            out.append("Sales: some dates cannot be read")
        elif (shop.today - d.min()).days < 120:
            out.append("Sales: at least 4 months of daily sales are needed (a year or more is better)")
        q = pd.to_numeric(shop.sales["Qty"], errors="coerce")
        if q.isna().any() or (q < 0).any():
            out.append("Sales: 'Qty' must be a number of 0 or more (returns: leave them out)")
        extra = sorted(set(shop.sales["SKU"].astype(str)) - set(p["SKU"].astype(str)))
        if extra:
            out.append(f"Sales: {len(extra)} SKU(s) not in Products, e.g. {extra[0]} (they are ignored)")
    return out


def blocking(problems):
    return [x for x in problems if "ignored" not in x]


# ------------------------------------------------------------------------------------------------ Excel
def to_excel(shop: Shop) -> bytes:
    bio = io.BytesIO()
    settings = pd.DataFrame({"Setting": ["Shop", "Currency", "Today", "Budget", "Target fill rate"],
                             "Value": [shop.name, shop.currency, shop.today.date().isoformat(),
                                       "" if shop.budget is None else shop.budget, shop.target]})
    sup = shop.suppliers.copy()
    sup["Next order date"] = pd.to_datetime(sup["Next order date"]).dt.date
    sales = shop.sales.copy()
    sales["Date"] = pd.to_datetime(sales["Date"]).dt.date
    ev = shop.events.copy()
    if len(ev):
        ev["Date"] = pd.to_datetime(ev["Date"]).dt.date
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        settings.to_excel(xw, sheet_name="Settings", index=False)
        shop.products[PRODUCT_COLS].to_excel(xw, sheet_name="Products", index=False)
        sup[SUPPLIER_COLS].to_excel(xw, sheet_name="Suppliers", index=False)
        sales[["Date", "SKU", "Qty"]].to_excel(xw, sheet_name="Sales", index=False)
        ev.to_excel(xw, sheet_name="Events", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(11, min(42, max(
                    len(str(c.value or "")) for c in col[:200]) + 2))
    return bio.getvalue()


def from_excel(data) -> Shop:
    xl = pd.ExcelFile(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    need = {"Products", "Suppliers", "Sales"}
    if not need <= set(xl.sheet_names):
        raise ValueError(f"the workbook needs the sheets {', '.join(sorted(need))}")
    kv = {}
    if "Settings" in xl.sheet_names:
        st = pd.read_excel(xl, "Settings")
        kv = {str(a).strip().lower(): b for a, b in zip(st.iloc[:, 0], st.iloc[:, 1])}
    products = pd.read_excel(xl, "Products")
    products["SKU"] = products["SKU"].astype(str)
    suppliers = pd.read_excel(xl, "Suppliers")
    sales = pd.read_excel(xl, "Sales")
    sales["SKU"] = sales["SKU"].astype(str)
    events = pd.read_excel(xl, "Events") if "Events" in xl.sheet_names else pd.DataFrame(columns=["Date", "Event"])
    today = kv.get("today")
    today = pd.Timestamp(today).normalize() if today is not None and str(today) not in ("", "nan") else \
        pd.to_datetime(sales["Date"]).max().normalize() + pd.Timedelta(days=1)
    budget = kv.get("budget")
    try:
        budget = float(budget) if budget is not None and str(budget).strip() not in ("", "nan") else None
    except ValueError:
        budget = None
    try:
        target = float(kv.get("target fill rate", 0.95))
        target = target / 100 if target > 1 else target
    except (TypeError, ValueError):
        target = 0.95
    if "Next order date" in suppliers.columns:
        suppliers["Next order date"] = pd.to_datetime(suppliers["Next order date"], errors="coerce").fillna(today)
    return Shop(str(kv.get("shop", "My shop")), str(kv.get("currency", "Rs")) if str(kv.get("currency", "")) not in
                ("", "nan") else "Rs", today, products, suppliers, sales, events, budget, target)


# ------------------------------------------------------------------------------------------------ demo
CATALOGUE = {  # category: (supplier, [(product, price Rs, base units/day)])
    "Pain & fever": ("MedLine Distributors", [
        ("Paracetamol 500 mg, strip of 10", 45, 22), ("Paracetamol syrup for children 60 ml", 180, 4),
        ("Ibuprofen 400 mg, strip of 10", 95, 7), ("Aspirin 75 mg, strip of 14", 70, 5),
        ("Pain relief gel 30 g", 650, 1.5), ("Muscle rub balm 50 g", 420, 2.2), ("Migraine relief tablets, 12", 540, 0.8),
        ("Menthol inhaler", 160, 1.2), ("Hot and cold pack", 950, 0.3), ("Paracetamol 650 mg, strip of 10", 75, 3)]),
    "Cold & flu": ("MedLine Distributors", [
        ("Cough syrup 100 ml", 390, 6), ("Dry cough syrup 100 ml", 450, 3.5), ("Throat lozenges, 16", 280, 5),
        ("Nasal spray 15 ml", 720, 1.8), ("Steam inhalation capsules, 10", 260, 1.4), ("Cold and flu sachets, 10", 590, 3),
        ("Saline nasal drops 10 ml", 240, 1.6), ("Vapour rub 50 g", 480, 2.4), ("Antihistamine tablets, 10", 210, 4),
        ("Children's cough syrup 60 ml", 340, 2)]),
    "Stomach": ("HealthCare Wholesale", [
        ("Oral rehydration salts, sachet", 35, 14), ("Antacid tablets, 20", 160, 4), ("Antacid liquid 200 ml", 520, 1.6),
        ("Probiotic capsules, 10", 890, 0.9), ("Anti-diarrhoea tablets, 6", 140, 2.5), ("Laxative syrup 100 ml", 610, 0.9),
        ("Digestive enzyme tablets, 15", 330, 1.1), ("Gripe water 120 ml", 290, 1.8), ("Peppermint oil capsules, 20", 940, 0.4),
        ("Glucose powder 400 g", 470, 2.7)]),
    "Vitamins": ("Island Pharma Imports", [
        ("Vitamin C 500 mg, 30 tablets", 690, 3.2), ("Multivitamin, 30 tablets", 1250, 2.4),
        ("Vitamin D3 1000 IU, 60", 1480, 1.3), ("Calcium with vitamin D, 30", 980, 1.7), ("Iron supplement, 30", 560, 1.5),
        ("Zinc tablets, 30", 520, 1.1), ("Fish oil capsules, 60", 2150, 0.8), ("Folic acid 5 mg, 30", 260, 1.9),
        ("Children's multivitamin syrup 200 ml", 1150, 0.9), ("Vitamin B complex, 30", 380, 2.1),
        ("Magnesium tablets, 30", 1320, 0.35), ("Biotin tablets, 30", 1890, 0.25)]),
    "First aid": ("HealthCare Wholesale", [
        ("Adhesive bandages, box of 20", 240, 3.5), ("Antiseptic liquid 500 ml", 760, 1.9), ("Cotton wool 100 g", 210, 2.6),
        ("Gauze swabs, pack of 10", 180, 1.7), ("Crepe bandage 7.5 cm", 290, 1.1), ("Medical tape 2.5 cm", 150, 1.3),
        ("Antiseptic cream 30 g", 340, 2), ("Burn gel 25 g", 480, 0.6), ("Digital thermometer", 1650, 0.4),
        ("First aid kit, small", 2900, 0.12), ("Elastic knee support", 1850, 0.15), ("Instant ice pack", 380, 0.5)]),
    "Diabetes care": ("Island Pharma Imports", [
        ("Glucose test strips, 50", 4200, 1.4), ("Lancets, 100", 1350, 0.8), ("Glucometer", 5900, 0.08),
        ("Glucose tablets, 10", 220, 1.2), ("Diabetic foot cream 100 g", 1100, 0.3), ("Insulin pen needles, 100", 3800, 0.45),
        ("Sugar-free sweetener, 100 tablets", 650, 1)]),
    "Baby care": ("Baby & Beauty Supplies", [
        ("Baby diapers M, pack of 30", 2450, 2.3), ("Baby diapers L, pack of 28", 2550, 1.9),
        ("Baby wipes, 80", 690, 3.4), ("Nappy rash cream 50 g", 720, 1.2), ("Baby lotion 200 ml", 890, 0.9),
        ("Baby shampoo 200 ml", 780, 0.8), ("Feeding bottle 250 ml", 1450, 0.25), ("Teething gel 10 g", 590, 0.5),
        ("Baby powder 200 g", 520, 1), ("Baby oil 100 ml", 610, 0.6)]),
    "Skin & personal care": ("Baby & Beauty Supplies", [
        ("Sunscreen SPF 50, 100 ml", 1890, 0.9), ("Hand sanitiser 250 ml", 450, 3), ("Face masks, box of 50", 950, 1.6),
        ("Moisturising lotion 400 ml", 1350, 0.9), ("Lip balm", 240, 2), ("Mosquito repellent cream 50 g", 380, 3.4),
        ("Mosquito repellent spray 100 ml", 790, 1.5), ("Antifungal powder 100 g", 420, 1.1), ("Prickly heat powder 150 g", 310, 2.2),
        ("Hand cream 75 ml", 560, 0.6), ("Aloe vera gel 100 ml", 640, 0.5), ("Acne gel 15 g", 520, 0.45)]),
}
SUPPLIERS = [("MedLine Distributors", 2, 7, 15000), ("HealthCare Wholesale", 3, 7, 10000),
             ("Island Pharma Imports", 14, 14, 60000), ("Baby & Beauty Supplies", 5, 14, 8000)]
PACKS = {"MedLine Distributors": (10, 12, 20), "HealthCare Wholesale": (6, 10, 12, 24),
         "Island Pharma Imports": (6, 10, 12), "Baby & Beauty Supplies": (4, 6, 12)}


def _season(cat, doy):
    """Fictional yearly patterns: two monsoon seasons for colds, fevers and mosquitoes, hot months for skin care."""
    x = 2 * np.pi * doy / 365.25
    mons = np.exp(-((doy - 175) / 28) ** 2) + 0.8 * np.exp(-((doy - 320) / 30) ** 2)       # June, mid-November
    hot = np.exp(-((doy - 95) / 35) ** 2)                                                    # March-April
    return {"Cold & flu": 1 + 0.9 * mons, "Pain & fever": 1 + 0.35 * mons,
            "Stomach": 1 + 0.25 * np.sin(x - 1.0) + 0.2 * mons,
            "Skin & personal care": 1 + 0.5 * hot + 0.2 * mons,
            "Vitamins": 1 + 0.2 * mons}.get(cat, 1 + 0.05 * np.sin(x))


def demo(today="2026-11-02", seed=3) -> Shop:
    """Lakeside Pharmacy (fictional): 83 products, 4 suppliers, two years of daily sales up to yesterday."""
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today)
    days = pd.date_range(today - pd.Timedelta(days=730), today - pd.Timedelta(days=1))
    doy = days.dayofyear.to_numpy()
    wd = days.weekday.to_numpy()
    week = np.array([1.05, 1.0, 1.0, 1.02, 1.1, 1.18, 0.65])                  # Mon..Sun (closed Sunday afternoon)
    events = []
    for y in (days[0].year, days[0].year + 1, days[0].year + 2):
        for m, d, name, lift in ((4, 13, "New Year festival", 0.55), (4, 14, "New Year festival", 0.5),
                                 (12, 25, "Christmas", 0.6), (12, 31, "Year end", 0.2)):
            dt = pd.Timestamp(year=y, month=m, day=d)
            if days[0] <= dt <= days[-1] + pd.Timedelta(days=60):
                events.append((dt, name, lift))
        for k in range(12):                                              # full-moon holidays: shops shut early
            dt = pd.Timestamp(year=y, month=1, day=13) + pd.Timedelta(days=round(29.53 * k))
            if days[0] <= dt <= days[-1] + pd.Timedelta(days=60):
                events.append((dt, "Full moon holiday", -0.35))
    ev_effect = np.zeros(len(days))
    for dt, _, lift in events:
        if dt in days:
            ev_effect[days.get_loc(dt)] += lift
        if lift > 0 and dt - pd.Timedelta(days=1) in days:                # stocking up the day before
            ev_effect[days.get_loc(dt - pd.Timedelta(days=1))] += lift * 0.6
    rows, prods = [], []
    k = 0
    for cat, (sup, items) in CATALOGUE.items():
        for name, price, base in items:
            k += 1
            sku = f"LP{k:03d}"
            trend = rng.choice([1.0, 1.0, 1.0, 1.25, 0.8, 1.5, 0.55], p=[.3, .2, .15, .15, .1, .05, .05])
            t = np.linspace(0, 1, len(days))
            lvl = base * (1 + (trend - 1) * t) * rng.lognormal(0, 0.15)
            if trend == 1.5:                                             # a new product that took off
                lvl = lvl * np.clip((t - 0.25) * 2.5, 0.05, 1)
            mu = lvl * _season(cat, doy) * week[wd] * (1 + ev_effect)
            promo = np.zeros(len(days))
            for _ in range(rng.integers(0, 3)):                          # a few promotion weeks
                s0 = rng.integers(0, len(days) - 7)
                promo[s0:s0 + 7] = 0.7
            mu = mu * (1 + promo)
            disp = 0.15 + 0.35 * rng.random()
            y = rng.poisson(rng.gamma(1 / disp, mu * disp))
            nz = np.flatnonzero(y)
            rows += [(days[i], sku, int(y[i])) for i in nz]
            margin = rng.uniform(0.2, 0.35)
            avg = y[-28:].mean() if y[-28:].sum() else base
            pack = 1 if price > 1500 or base < 0.5 else int(rng.choice(PACKS[sup]))
            state = rng.choice(["ok", "low", "over", "dead"], p=[.55, .2, .2, .05])
            cover = {"ok": rng.uniform(10, 25), "low": rng.uniform(0, 5), "over": rng.uniform(45, 90),
                     "dead": 0}[state]
            on_hand = int(round(avg * cover)) if state != "dead" else int(rng.integers(20, 60))
            on_order = int(pack * rng.integers(1, 4)) if rng.random() < 0.15 else 0
            moq = pack if (pack == 1 or rng.random() < 0.7) else pack * int(rng.integers(2, 4))
            prods.append((sku, name, cat, sup, round(price * (1 - margin)), price, on_hand, on_order, pack, moq))
    products = pd.DataFrame(prods, columns=PRODUCT_COLS)
    # dead stock: one product stopped selling half a year ago
    sales = pd.DataFrame(rows, columns=["Date", "SKU", "Qty"])
    dead = products.loc[products["Category"] == "Skin & personal care", "SKU"].iloc[-2]
    sales = sales[~((sales["SKU"] == dead) & (sales["Date"] > today - pd.Timedelta(days=200)))]
    sup_rows = []
    for name, lead, every, mn in SUPPLIERS:
        nxt = today + pd.Timedelta(days=int(rng.integers(0, min(every, 3))))
        sup_rows.append((name, lead, every, nxt, mn))
    suppliers = pd.DataFrame(sup_rows, columns=SUPPLIER_COLS)
    suppliers.loc[suppliers["Supplier"] == "MedLine Distributors", "Next order date"] = today
    suppliers.loc[suppliers["Supplier"] == "Island Pharma Imports", "Next order date"] = today
    ev = pd.DataFrame([(d.date(), n) for d, n, _ in events], columns=["Date", "Event"]).sort_values("Date")
    return Shop("Lakeside Pharmacy (fictional)", "Rs", today, products, suppliers,
                sales.reset_index(drop=True), ev.reset_index(drop=True), None, 0.95)


__all__ = ["Shop", "check", "blocking", "to_excel", "from_excel", "demo", "math"]
