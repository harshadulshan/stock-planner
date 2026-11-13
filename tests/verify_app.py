"""
App checks: clicks through every tab of the Streamlit app (Streamlit's AppTest, no browser) with the demo pharmacy:
data, stock health, forecast, today's orders with and without a budget, the replay, the trade-off and the benchmark.

    python tests/verify_app.py     # about 2 minutes
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

passed, failed = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def metric(at, label):
    m = [m for m in at.metric if m.label == label]
    return m[0] if m else None


def money(v):
    return float(str(v).replace("Rs", "").replace(",", "").strip())


def errors(at):
    return [str(e.value)[:300] for e in at.exception]


at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
at.run()
ok("app starts without errors", not errors(at), str(errors(at)))
ok("eight tabs", [t.label for t in at.tabs][:8] == ["📥 Data", "🏪 Stock health", "🔮 Forecast", "🛒 Order today",
                                                    "🎬 Replay", "⚖️ Trade-off", "🧪 Benchmark", "ℹ️ How it works"])
ok("before a shop is loaded: ask to load one", any("Load the demo" in i.value for i in at.info))

button(at, "Load the demo pharmacy").click().run()
ok("demo loads and forecasts without errors", not errors(at), str(errors(at)))
ok("83 products, 4 suppliers, 730 days of sales", metric(at, "Products").value == "83" and
   metric(at, "Suppliers").value == "4" and metric(at, "Days of sales history").value == "730")
ok("data check passed", any("The data is complete" in s.value for s in at.success))
fe = metric(at, "Weekly forecast error, all products")
ok("forecast error shown, better than the 4-week average", fe is not None and str(fe.delta).startswith("-"),
   f"{fe.value if fe else None} {fe.delta if fe else None}")
med, isl = metric(at, "MedLine Distributors"), metric(at, "Island Pharma Imports")
ok("today's order day: MedLine and Island Pharma, each above its minimum or waiting",
   med is not None and isl is not None and money(med.value) >= 15000 and money(isl.value) >= 60000,
   f"{med.value if med else None} {isl.value if isl else None}")
labels = [d.label for d in at.get("download_button")]
ok("purchase orders can be downloaded", any("Purchase orders" in x for x in labels), str(labels))
total_free = money(med.value) + money(isl.value)

at.toggle(key="use_budget").set_value(True).run()
at.number_input(key="budget").set_value(60000).run()
med2, isl2 = metric(at, "MedLine Distributors"), metric(at, "Island Pharma Imports")
spent = money(med2.value) + money(isl2.value)
ok("a budget of Rs 60,000 is never exceeded", spent <= 60000 + 1e-6 and not errors(at), f"{spent}")
ok("with the budget, orders below a supplier's minimum wait",
   all(money(m.value) == 0 or money(m.value) >= mn for m, mn in ((med2, 15000), (isl2, 60000))),
   f"{med2.value} {isl2.value}")
at.toggle(key="use_budget").set_value(False).run()
ok("without the budget the full orders come back",
   abs(money(metric(at, "MedLine Distributors").value) + money(metric(at, "Island Pharma Imports").value)
       - total_free) < 1e-6)

v0 = money(metric(at, "At your target").value)
at.slider(key="target").set_value(98).run()
v1 = money(metric(at, "At your target").value)
ok("a higher service target needs more stock", v1 > v0, f"{v0} -> {v1}")
at.slider(key="target").set_value(95).run()

button(at, "▶ Replay").click().run()
ok("replay runs without errors", not errors(at), str(errors(at)))
served = metric(at, "Demand served from the shelf")
stock = metric(at, "Average stock value")
ok("replay: Stock Planner serves more of the demand than the rule of thumb",
   served is not None and str(served.delta).startswith("+"), f"{served.value if served else None} "
                                                             f"{served.delta if served else None}")
ok("replay: with less stock", stock is not None and str(stock.delta).startswith("-"),
   f"{stock.value if stock else None} {stock.delta if stock else None}")
at.slider(key="low_days").set_value(21).run()
ok("changing the rule asks for a new replay", any("settings changed" in w.value for w in at.warning))

at.selectbox(key="pick").set_value(5).run()
ok("another product's forecast", not errors(at) and metric(at, "On hand / on order") is not None)
ok("benchmark tab renders", not errors(at))
ok("how it works explains the method", any("lost sale" in m.value for m in at.markdown))

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
