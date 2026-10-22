"""Подразумеваемый лаг и подразумеваемая маржа по раундам и котировкам.

Для каждого события из data/inputs/events.csv:
  L_obs  — лидерство лаборатории над лучшей открытой моделью на дату,
           (ECI лаборатории − ECI лучшей открытой) / скорость фронтира;
  GM*    — долгосрочная валовая маржа, при которой DCF = цене;
  L_impl — лаг, при котором DCF = цене, на кривой GM(L) из dcf.py.
Рост у всех общий (g0, tau): всё, что рынок думает о росте иначе,
уходит в подразумеваемую величину — как в подразумеваемую волатильность.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcf import Firm, _value, implied_lag, value  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ECI = ROOT / "data/epoch_eci_scores_2026-10-22.csv"
EVENTS = ROOT / "data/inputs/events.csv"
OUT = ROOT / "data/derived"
V_FRONT = 1.43
GM_TARGET_CLOSED = 0.75


def lead_months(e, org, date):
    if not isinstance(org, str) or not org:
        return float("nan")
    past = e[e.date <= date]
    lab = past.loc[past.Organization == org, "eci"]
    opn = past.loc[past["Accessibility group"] == "Open weights", "eci"]
    if not len(lab):
        return float("nan")
    return (lab.max() - opn.max()) / V_FRONT


def firm_for(r, g0, tau, wacc, gm_target, l_now):
    return Firm(r.firm, rev0=r.rr_usd_b, g0=g0, gm_now=r.gm_now,
                gm_target=gm_target, l_now=l_now, cash=r.cash_usd_b,
                tau=tau, wacc=wacc)


def implied_gm(r, g0, tau, wacc):
    """GM* при L = L_now = 0: кривая сводится к GM ≡ gm_target.
    Без опции выхода в хостинг: иначе её пол в 25 % маскирует ответ."""
    lo, hi = 0.0, 0.85
    f = lambda gm: _value(firm_for(r, g0, tau, wacc, gm, 0.0), 0.0, True)["equity"]
    if f(hi) < r.value_usd_b:
        return float("nan")      # не достижимо даже при потолке маржи
    if f(lo) > r.value_usd_b:
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < r.value_usd_b else (lo, mid)
    return (lo + hi) / 2


def run(g0=1.0, tau=2.5, wacc=0.12):
    e = pd.read_csv(ECI, parse_dates=["date"])
    ev = pd.read_csv(EVENTS, parse_dates=["date"])
    rows = []
    for r in ev.itertuples():
        L_obs = lead_months(e, r.eci_org, r.date)
        L_cal = max(L_obs, 0.0) if L_obs == L_obs else 0.0
        f = firm_for(r, g0, tau, wacc, GM_TARGET_CLOSED, L_cal)
        rows.append(dict(
            firm=r.firm, date=r.date.date(), event=r.event, value=r.value_usd_b,
            rr=r.rr_usd_b, mult=r.value_usd_b / r.rr_usd_b, L_obs=L_obs,
            dcf_at_obs=value(f, L_cal)["equity"],
            L_impl=implied_lag(f, r.value_usd_b),
            gm_star=implied_gm(r, g0, tau, wacc), gm_now=r.gm_now, conf=r.conf))
    return pd.DataFrame(rows)


def g0_for_path(mult, years, tau=2.5, tg=0.03):
    """g0, при котором выручка через years лет = mult × нынешней."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        g0 = (lo + hi) / 2
        m = 1.0
        for i in range(years):
            m *= 1 + tg + (g0 - tg) * __import__("math").exp(-i / tau)
        lo, hi = (g0, hi) if m < mult else (lo, g0)
    return (lo + hi) / 2


# независимые прогнозы выручки: (фирма, дата события) -> (кратность, лет, источник)
ANCHORS = {
    ("Anthropic", "2026-10-15"): (195 / 67, 2, "банкиры IPO: выручка 2028 г. $190–200 млрд (Reuters, авг. 2026)"),
    ("OpenAI", "2026-03-31"): (280 / 24, 4, "план компании: выручка 2030 г. >$280 млрд (CNBC/Reuters, фев. 2026)"),
}


def anchored(wacc=0.12):
    e = pd.read_csv(ECI, parse_dates=["date"])
    ev = pd.read_csv(EVENTS, parse_dates=["date"])
    rows = []
    for (firm, d), (mult, yrs, src) in ANCHORS.items():
        r = ev[(ev.firm == firm) & (ev.date == d)].iloc[0]
        g0 = g0_for_path(mult, yrs)
        L_obs = lead_months(e, r.eci_org, r.date)
        f = firm_for(r, g0, 2.5, wacc, GM_TARGET_CLOSED, max(L_obs, 0.0))
        v_obs = value(f, L_obs)["equity"]
        dv = (value(f, L_obs + 1)["equity"] - value(f, L_obs - 1)["equity"]) / 2
        rows.append(dict(firm=firm, date=d, value=r.value_usd_b, g0=g0,
                         L_obs=L_obs, dcf_at_obs=v_obs, L_impl=implied_lag(f, r.value_usd_b),
                         dv_per_month=dv, dv_pct=dv / v_obs, source=src))
    return pd.DataFrame(rows)


def iso_curve(firm, d, g0s=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0), wacc=0.12):
    """Пары (g0, L), совместимые с ценой события: лаг не отделим от роста."""
    e = pd.read_csv(ECI, parse_dates=["date"])
    ev = pd.read_csv(EVENTS, parse_dates=["date"])
    r = ev[(ev.firm == firm) & (ev.date == d)].iloc[0]
    L_obs = max(lead_months(e, r.eci_org, r.date), 0.0)
    out = {}
    for g0 in g0s:
        f = firm_for(r, g0, 2.5, wacc, GM_TARGET_CLOSED, L_obs)
        out[g0] = implied_lag(f, r.value_usd_b)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    print("Якорь на независимый прогноз выручки:")
    a = anchored()
    a.to_csv(OUT / "implied_anchored.csv", index=False)
    print(a.drop(columns=["source"]).round(2).to_string(index=False))
    for s in a.source:
        print("  ·", s)
    print("\nИзо-кривые (g0 → подразумеваемый L, мес; None — не достижимо при GM ≤ 85 %):")
    for firm, d in [("Anthropic", "2026-05-28"), ("Anthropic", "2026-10-15"),
                    ("OpenAI", "2026-03-31"), ("Mistral", "2026-09-08"),
                    ("Moonshot", "2026-10-06"), ("Zhipu", "2026-10-16")]:
        c = iso_curve(firm, d)
        print(f"  {firm:9s} {d}: " + "  ".join(
            f"{int(g * 100)}%→{'—' if v is None else f'{v:.1f}'}" for g, v in c.items()))
    print()
    pd.set_option("display.width", 200)
    base = run()
    base.to_csv(OUT / "implied_base.csv", index=False)
    show = base.copy()
    for c in ["mult", "L_obs", "L_impl"]:
        show[c] = show[c].astype(float).round(1)
    show["dcf_at_obs"] = show["dcf_at_obs"].round(0)
    for c in ["gm_star", "gm_now"]:
        show[c] = (show[c] * 100).round(0)
    print("База: g0=100 %, tau=2,5 г., WACC 12 %, долгосрочная GM закрытых 75 % при L_obs")
    print(show.drop(columns=["conf"]).to_string(index=False))

    print("\nЧувствительность подразумеваемого лага (мес) к росту и ставке:")
    grid = {"g0 70 %": dict(g0=0.7), "g0 150 %": dict(g0=1.5),
            "WACC 10 %": dict(wacc=0.10), "WACC 14 %": dict(wacc=0.14)}
    sens = base[["firm", "date", "event"]].copy()
    sens["база"] = base.L_impl
    for k, kw in grid.items():
        sens[k] = run(**kw).L_impl
    print(sens.round(1).to_string(index=False))
    sens.to_csv(OUT / "implied_sensitivity.csv", index=False)
