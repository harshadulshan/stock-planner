"""Files for the shop: purchase orders per supplier (Excel), the stock health list."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def purchase_orders(orders: pd.DataFrame, shop) -> bytes:
    """One sheet per supplier ordering today (product, quantity, packs, value), and a summary sheet."""
    from openpyxl.styles import Font, PatternFill

    bio = io.BytesIO()
    today = orders[orders["today"] & (orders["qty"] > 0)]
    sup = shop.suppliers.set_index("Supplier")
    rows = []
    for s, g in today.groupby("supplier"):
        rows.append({"Supplier": s, "Lines": len(g), "Value": float(g["value"].sum()),
                     "Minimum order": float(sup.loc[s, "Min order value"]) if s in sup.index else 0.0,
                     "Delivery expected": (shop.today + pd.Timedelta(days=int(sup.loc[s, "Lead time (days)"])))
                     .date().isoformat() if s in sup.index else ""})
    summary = pd.DataFrame(rows, columns=["Supplier", "Lines", "Value", "Minimum order", "Delivery expected"])
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        head = pd.DataFrame({"": [shop.name, f"Orders for {shop.today.date().isoformat()}",
                                  f"Total {shop.currency} {summary['Value'].sum():,.0f}"]})
        head.to_excel(xw, sheet_name="Summary", index=False, header=False)
        summary.to_excel(xw, sheet_name="Summary", index=False, startrow=4)
        for s, g in today.groupby("supplier"):
            lines = pd.DataFrame({"SKU": g["SKU"], "Product": g["Product"], "Quantity": g["qty"].astype(int),
                                  "Packs": (g["qty"] / g["pack"]).round(1), "Pack size": g["pack"],
                                  "Unit cost": g["unit_cost"], "Value": g["value"], "Note": g["note"]})
            name = "".join(ch for ch in str(s) if ch not in '[]:*?/\\')[:31]
            lines.to_excel(xw, sheet_name=name, index=False)
        fill = PatternFill("solid", fgColor="DCE9F9")
        for ws in xw.book.worksheets:
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
                for c in row:
                    if ws.title != "Summary" and c.row == 1:
                        c.font = Font(bold=True)
                        c.fill = fill
                    if ws.title == "Summary" and c.row == 5:
                        c.font = Font(bold=True)
                        c.fill = fill
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(10, min(48, max(len(str(c.value or ""))
                                                                                       for c in col) + 2))
    return bio.getvalue()


def health_table(shop, classes: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """One row per product: class, sales, stock, days of cover, risk before the next delivery."""
    p = shop.products.reset_index(drop=True)
    cover = classes["days_of_cover"].to_numpy(float)
    status = np.where(classes["last_sale_days"].to_numpy() >= 60, "no sale for 2 months",
                      np.where(orders["risk_before_arrival"].to_numpy(float) >= 0.2, "may run out",
                               np.where(cover > 90, "too much stock", "fine")))
    return pd.DataFrame({"SKU": p["SKU"], "Product": p["Product"], "Category": p["Category"],
                         "Class": classes["ABC"] + classes["XYZ"], "Sold a day": classes["avg_per_day"].round(2),
                         "On hand": p["On hand"], "Days of stock": np.where(np.isfinite(cover), cover.round(0), np.nan),
                         "Stock value": classes["stock_value"].round(0),
                         "Risk of running out": orders["risk_before_arrival"].astype(float),
                         "Status": status})
