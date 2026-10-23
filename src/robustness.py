"""Проверки после рецензии 2026-10-22: разложение наклона, интервалы для L,
таблица чувствительности. Пишет paper/tables/{decomp,lead_ci,sensitivity}.tex.

  python -P src/robustness.py
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcf  # noqa: E402
import implied  # noqa: E402
import paper_outputs as po  # noqa: E402
import price_premium  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "paper/tables"
V = price_premium.V_FRONT
SEED = 20261023


# ---------- 1. наклон: издержки или рента ----------
def decomposition(n_boot=4000):
    m = price_premium.load()
    r = m[m.date >= "2025-07-01"].copy()
    groups = r["org"].astype("category").cat.codes
    hc = smf.ols("lp ~ eci * open", r).fit(cov_type="HC1")
    cl = smf.ols("lp ~ eci * open", r).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b_c, b_x = hc.params["eci"], hc.params["eci:open"]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_boot):
        s = r.sample(len(r), replace=True, random_state=rng.integers(1 << 31))
        if s.open.nunique() < 2:
            continue
        f = smf.ols("lp ~ eci * open", s).fit()
        draws.append((f.params["eci"], f.params["eci"] + f.params["eci:open"], -f.params["eci:open"]))
    d = np.array(draws)
    p5, p95 = np.percentile(d, [5, 95], axis=0)
    rows = [
        ("Closed weights", b_c, hc.bse["eci"], cl.bse["eci"], p5[0], p95[0]),
        ("Open weights ($\\approx$ cost)", b_c + b_x, None, None, p5[1], p95[1]),
        ("Rent (closed $-$ open)", -b_x, hc.bse["eci:open"], cl.bse["eci:open"], p5[2], p95[2]),
    ]
    lines = []
    for name, b, se, sec, lo, hi in rows:
        se_t = "---" if se is None else f"{se:.3f}"
        sec_t = "---" if sec is None else f"{sec:.3f}"
        lam = f"{1 / (b * V):.1f}" if b > 0 else "$\\infty$"
        lines.append(f"{name} & {b:.3f} & {se_t} & {sec_t} & [{lo:.3f}; {hi:.3f}] & {lam} \\\\")
    (TAB / "decomp.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = dict(b_closed=b_c, b_open=b_c + b_x, rent=-b_x, rent_se=hc.bse["eci:open"],
               rent_p5=p5[2], rent_p95=p95[2], p_rent_le0=float((d[:, 2] <= 0).mean()),
               lam_rent_bound=1 / (p95[2] * V), n_clusters=int(r.org.nunique()))
    return out


# ---------- 2. интервалы для L ----------
def lead_draws(n=2000):
    """ECI каждой модели — расщеплённое нормальное по 90%-интервалу Epoch;
    модели независимы (общая шкала индекса не учитывается)."""
    e = pd.read_csv(implied.ECI, parse_dates=["date"])
    z = 1.6449
    sd_lo = (e.eci - e.eci_ci_low).clip(lower=1e-6) / z
    sd_hi = (e.eci_ci_high - e.eci).clip(lower=1e-6) / z
    rng = np.random.default_rng(SEED)
    for _ in range(n):
        g = np.abs(rng.standard_normal(len(e)))
        up = rng.random(len(e)) < sd_hi / (sd_lo + sd_hi)
        x = e.copy()
        x["eci"] = np.where(up, e.eci + g * sd_hi, e.eci - g * sd_lo)
        yield x


def lead_ci(n=2000):
    ev = pd.read_csv(implied.EVENTS, parse_dates=["date"])
    ev = ev[ev.firm.isin(["Anthropic", "OpenAI"])]
    e0 = pd.read_csv(implied.ECI, parse_dates=["date"])
    point = [implied.lead_months(e0, r.eci_org, r.date) for r in ev.itertuples()]
    sims = np.array([[implied.lead_months(x, r.eci_org, r.date) for r in ev.itertuples()]
                     for x in lead_draws(n)])
    lo, hi = np.percentile(sims, [5, 95], axis=0)
    lines = [f"{r.firm} & {r.date:%Y-%m-%d} & {r.event} & ${p:.1f}$ & [${a:.1f}$; ${b:.1f}$] \\\\"
             for r, p, a, b in zip(ev.itertuples(), point, lo, hi)]
    (TAB / "lead_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame(dict(firm=ev.firm.values, date=ev.date.values, L=point, lo=lo, hi=hi))


# ---------- 3. чувствительность ----------
BASE = dict(lam=dcf.LAM, eps=dcf.EPS, cap=dcf.GM_CAP, reinv=dcf.REINV,
            gm_ref=po.GM_REF, l_ref=po.L_REF, growth=1.0)


@contextmanager
def setting(**kw):
    p = {**BASE, **kw}
    saved = (dcf.LAM, dcf.EPS, dcf.GM_CAP, dcf.REINV, po.GM_REF, po.L_REF)
    dcf.LAM, dcf.EPS, dcf.GM_CAP, dcf.REINV = p["lam"], p["eps"], p["cap"], p["reinv"]
    po.GM_REF, po.L_REF = p["gm_ref"], p["l_ref"]
    try:
        yield p
    finally:
        dcf.LAM, dcf.EPS, dcf.GM_CAP, dcf.REINV, po.GM_REF, po.L_REF = saved


def anchored(firm, d, growth):
    mult, yrs, _ = implied.ANCHORS[(firm, d)]
    r = po.row(firm, d)
    f = po.firm_at(r, implied.g0_for_path(mult * growth, yrs))
    L0 = po.l_obs(r)
    v0 = dcf.value(f, L0)["equity"]
    dv = (dcf.value(f, L0 + 1)["equity"] - dcf.value(f, L0 - 1)["equity"]) / 2
    return v0, implied.implied_lag(f, r.value_usd_b), dv / v0


CASES = [
    ("Baseline", {}),
    ("$\\lambda = 3.6$ (vintage)", dict(lam=3.6)),
    ("$\\lambda = 11.0$ (bootstrap p95)", dict(lam=11.0)),
    ("$\\lambda = 54$ (rent, point est.)", dict(lam=54.0)),
    ("$\\varepsilon = 0.5$", dict(eps=0.5)),
    ("$\\varepsilon = 1.33$ (Lerner)", dict(eps=1.33)),
    ("Margin cap 92\\%", dict(cap=0.92)),
    ("Margin cap 99\\%", dict(cap=0.99)),
    ("Margin 77\\% at $L = 1.3$ (Nov 2025)", dict(gm_ref=0.77, l_ref=1.3)),
    ("Margin 65\\% at $L = 6$", dict(gm_ref=0.65)),
    ("Growth anchor $-20\\%$", dict(growth=0.8)),
    ("Growth anchor $+20\\%$", dict(growth=1.2)),
    ("Reinvestment 50\\%", dict(reinv=0.50)),
    ("Reinvestment 100\\%", dict(reinv=1.00)),
]


def sensitivity():
    rows, lines = [], []
    for name, kw in CASES:
        with setting(**kw) as p:
            a = anchored("Anthropic", "2026-10-15", p["growth"])
            o = anchored("OpenAI", "2026-03-31", p["growth"])
        rows.append(dict(case=name, a_v=a[0], a_L=a[1], a_el=a[2], o_v=o[0], o_L=o[1], o_el=o[2]))
        L = lambda x: "unatt." if x is None else f"{x:.1f}"
        lines.append(f"{name} & {a[0] / 1000:.2f} & {L(a[1])} & {a[2] * 100:.0f} & "
                     f"{o[0] / 1000:.2f} & {L(o[1])} & {o[2] * 100:.0f} \\\\")
    (TAB / "sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    dec = decomposition()
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in dec.items()})
    print(lead_ci().round(1).to_string(index=False))
    print(sensitivity().round(3).to_string(index=False))
