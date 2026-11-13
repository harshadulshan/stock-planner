"""
Ranges around the forecast: a negative binomial distribution with the forecast as its mean.

How wide the range is comes from a held-out period (sales the model had not seen): for windows of 1, 7, 14 and 28
days, the extra variance beyond Poisson is measured as phi in Var = mean + phi * mean^2, pooled over all products
(Poisson already makes slow sellers relatively more variable; measuring phi separately for slow and fast sellers
followed the quirks of one period and placed stock worse on M5). Each window length gets its own phi: day-to-day noise
partly cancels over a week, but when the model misjudges a product's level it is off on every day of the window.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

WINDOWS = (1, 7, 14, 28)
BANDS = (0.0, 0.2, 0.5, 1.0, 3.0, 10.0, np.inf)          # mean units a day


def band(mean_per_day):
    return np.clip(np.searchsorted(BANDS, mean_per_day, side="right") - 1, 0, len(BANDS) - 2)


class Dispersion:
    """phi per (window length, sales-rate band), learnt from forecasts and what actually sold."""

    def __init__(self, table=None, overall=None):
        self.table = table if table is not None else np.zeros((len(WINDOWS), len(BANDS) - 1))
        self.overall = overall if overall is not None else np.zeros(len(WINDOWS))

    @classmethod
    def fit(cls, mean, actual, shrink=1e12):
        """mean, actual: (n, days) forecasts and actual sales for a held-out period. shrink: how strongly each
        sales-rate band is pulled to the pooled value (the default pools completely)."""
        mean = np.asarray(mean, float)
        actual = np.asarray(actual, float)
        n, D = mean.shape
        b = band(mean.mean(axis=1))
        table = np.zeros((len(WINDOWS), len(BANDS) - 1))
        overall = np.zeros(len(WINDOWS))
        for k, w in enumerate(WINDOWS):
            if w > D:
                table[k] = table[k - 1]
                overall[k] = overall[k - 1]
                continue
            nw = D // w
            M = mean[:, :nw * w].reshape(n, nw, w).sum(axis=2)
            S = actual[:, :nw * w].reshape(n, nw, w).sum(axis=2)
            num = ((S - M) ** 2 - M).sum(axis=1)
            den = (M ** 2).sum(axis=1)
            overall[k] = max(0.0, num.sum() / max(den.sum(), 1e-9))
            for g in range(len(BANDS) - 1):
                sel = b == g
                cnt = sel.sum()
                if cnt == 0:
                    table[k, g] = overall[k]
                    continue
                ph = max(0.0, num[sel].sum() / max(den[sel].sum(), 1e-9))
                table[k, g] = (cnt * ph + shrink * overall[k]) / (cnt + shrink)
        return cls(table, overall)

    def phi(self, mean_per_day, window):
        """phi for products with these mean daily sales over a window of this many days (interpolated)."""
        mean_per_day = np.asarray(mean_per_day, float)
        w = np.clip(np.asarray(window, float), WINDOWS[0], WINDOWS[-1])
        b = band(mean_per_day)
        k = np.clip(np.searchsorted(WINDOWS, w, side="right") - 1, 0, len(WINDOWS) - 2)
        w0 = np.asarray(WINDOWS)[k]
        w1 = np.asarray(WINDOWS)[k + 1]
        t = np.clip((w - w0) / (w1 - w0), 0, 1)
        return (1 - t) * self.table[k, b] + t * self.table[k + 1, b]

    def to_dict(self):
        return {"windows": list(WINDOWS), "bands": [b if np.isfinite(b) else None for b in BANDS],
                "table": self.table.round(5).tolist(), "overall": self.overall.round(5).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["table"], float), np.array(d["overall"], float))


def nb(mean, phi):
    """scipy negative binomial with this mean and Var = mean + phi * mean^2 (Poisson when phi is 0)."""
    mean = np.maximum(np.asarray(mean, float), 1e-9)
    phi = np.maximum(np.asarray(phi, float), 1e-6)
    r = 1.0 / phi
    return stats.nbinom(r, r / (r + mean))


def quantiles(mean, phi, qs):
    """{q: units} for each quantile q, same shape as mean."""
    d = nb(mean, phi)
    return {q: np.where(np.asarray(mean) <= 1e-9, 0.0, d.ppf(q)) for q in qs}


def sample_paths(mu, phi_day, phi_window, size, rng):
    """Possible futures of one product: (size, days) daily sales. Each future has its own level (the forecast can be
    off for the whole period) on top of day-to-day noise, so that both the daily ranges (phi_day) and the range of
    the total over the period (phi_window) match what was measured."""
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    M, Q = mu.sum(), (mu ** 2).sum()
    phi_day = max(float(phi_day), 1e-6)
    v = max(0.0, (phi_window * M * M - phi_day * Q) / (M * M + phi_day * Q))
    level = rng.gamma(1.0 / v, v, size=(size, 1)) if v > 1e-6 else np.ones((size, 1))
    lam = rng.gamma(1.0 / phi_day, mu[None, :] * level * phi_day)
    return rng.poisson(lam)
