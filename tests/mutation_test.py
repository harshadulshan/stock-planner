"""
Mutation test: do the engine checks notice when the code is wrong?

Each mutant copies the package, plants one bug (a forecast that peeks at the future, an order that ignores lost
sales, a budget that is overspent, a scorer that is off ...) and runs `verify_engine.py` against the copy. Every
mutant must be caught.

    python tests/mutation_test.py            # about 50 minutes
    python tests/mutation_test.py budget     # only the mutants whose name contains "budget"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTANTS = [
    ("scorer counts the jump to the first sale", "m5.py",
     "    m = started[:, :-1]                      # changes from the first sale on (not the jump up to it)",
     "    m = started[:, 1:]"),
    ("scorer weights in the wrong order", "m5.py", "        self.weight = np.concatenate(wts)",
     "        self.weight = w[\"weight\"].to_numpy(np.float64)"),
    ("forecast peeks one day ahead", "engine.py",
     "        t = np.clip(days - horizon - k, 0, T - 1)\n        f[f\"lag_{k}\"] = np.where(days - horizon - k >= 0",
     "        t = np.clip(days - horizon - k + 1, 0, T - 1)\n        f[f\"lag_{k}\"] = np.where(days - horizon - k >= 0"),
    ("same weekday taken too recently", "engine.py", "    m0 = -(-horizon // 7)", "    m0 = horizon // 7"),
    ("range ignores the Poisson part", "uncertainty.py", "            num = ((S - M) ** 2 - M).sum(axis=1)",
     "            num = ((S - M) ** 2).sum(axis=1)"),
    ("order-up-to level one unit short", "policy.py",
     "        out[ii] = np.where(ok.any(axis=1), ok.argmax(axis=1), K)",
     "        out[ii] = np.maximum(np.where(ok.any(axis=1), ok.argmax(axis=1), K) - 1, 0)"),
    ("orders ignore lost sales", "policy.py", "        risky &= p_run_out > 1e-6", "        risky &= False"),
    ("minimum order quantity ignored", "policy.py",
     "    r = np.where(r > 0, np.maximum(r, np.ceil(moq / pack - 1e-9) * pack), 0)", "    r = np.where(r > 0, r, 0)"),
    ("budget overspent", "policy.py",
     "        if budget is None or float((qty * cost).sum()) <= budget + 1e-9:",
     "        if True:"),
    ("supplier minimum ignored", "policy.py",
     "            if val <= 0 or val >= mn - 1e-9:\n                continue",
     "            if True:\n                continue"),
    ("deliveries one day early", "simulate.py", "            when = d + lead[rows]",
     "            when = d + np.maximum(lead[rows] - 1, 0)"),
    ("empty shelf: sales wait instead of being lost", "simulate.py",
     "        out.lost[:, d] = demand[:, d] - s\n        oh -= s",
     "        out.lost[:, d] = demand[:, d] - s\n        oh -= demand[:, d]"),
    ("rule of thumb sees today's sales", "simulate.py",
     "        e = self.h0 + d                                     # days before the decision only",
     "        e = self.h0 + d + 1"),
    ("sales put on the wrong day", "shop.py", "(s[\"Date\"] - start).dt.days.to_numpy()",
     "np.minimum((s[\"Date\"] - start).dt.days.to_numpy() + 1, T - 1)"),
    ("replay model trained on the replayed weeks", "analysis.py",
     "    mod = engine.fit(a.hist, H, first_day=first, last_day=s0 - 1 - H, rounds=_rounds(n * 730))",
     "    mod = engine.fit(a.hist, H, first_day=first, last_day=T - 1, rounds=_rounds(n * 730))"),
    ("shop-wide orders ignore lost sales", "policy.py",
     "        r_[risky] = nb(m_l[jj], phi_l[jj]).sf(avail[jj] - 1) > 1e-6", "        r_[risky] = False"),
    ("shop-wide orders ignore the minimum order quantity", "policy.py",
     "    first = np.ceil(np.maximum(moq, 1) / pack).astype(int) * pack", "    first = pack.copy()"),
    ("latest weeks' correction sees the decision day's sales", "simulate.py",
     "            f = level_factor(act[rows, max(h0 + d - 28, 0):h0 + d], fc[rows, max(h0 + d - 28, 0):h0 + d],",
     "            f = level_factor(act[rows, max(h0 + d - 27, 0):h0 + d + 1], fc[rows, max(h0 + d - 27, 0):h0 + d + 1],"),
]


def main():
    only = sys.argv[1:]
    todo = [m for m in MUTANTS if not only or any(w in m[0] for w in only)]
    caught = 0
    for name, fname, old, new in todo:
        tmp = tempfile.mkdtemp(prefix="mutant_")
        try:
            shutil.copytree(os.path.join(ROOT, "stockplan"), os.path.join(tmp, "stockplan"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            if os.path.isdir(os.path.join(ROOT, "data_cache")):
                os.symlink(os.path.join(ROOT, "data_cache"), os.path.join(tmp, "data_cache"))
            path = os.path.join(tmp, "stockplan", fname)
            src = open(path).read()
            assert src.count(old) == 1, f"{name}: the text to mutate is not found exactly once"
            open(path, "w").write(src.replace(old, new))
            try:
                res = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tests", "verify_engine.py")],
                                     env=dict(os.environ, SP_ROOT=tmp), capture_output=True, text=True, timeout=1500)
            except subprocess.TimeoutExpired:
                print(f"  MISSED  {name:48s}  (checks did not finish)", flush=True)
                continue
            fails = [ln.strip()[6:] for ln in res.stdout.splitlines() if ln.strip().startswith("FAIL")]
            hit = res.returncode != 0
            caught += hit
            why = fails[0] if fails else (res.stderr.strip().splitlines() or ["crash"])[-1]
            print(f"  {'CAUGHT' if hit else 'MISSED'}  {name:48s}  {why[:90] if hit else ''}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{caught} of {len(todo)} mutants caught")
    sys.exit(0 if caught == len(todo) else 1)


if __name__ == "__main__":
    main()
