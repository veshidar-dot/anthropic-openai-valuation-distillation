"""Премия к цене за лидерство: гедоническая регрессия цены токена на ECI.

Цены — снимок OpenRouter 2026-10-22, ECI — Epoch. Открытые веса хостят
многие провайдеры, их цена ≈ предельные издержки; разрыв закрытой цены
и открытой при равном ECI — наценка, которую держит закрытость весов.

  log p = a + b·ECI + γ·open + ε
  λ = 1 / (b · v),  v — скорость фронтира (ECI/мес) из lag_series.py
  премия за L месяцев лидерства: exp(b·v·L) = exp(L/λ)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OR_JSON = ROOT / "data/raw/openrouter/models_2026-10-22.json"
ECI = ROOT / "data/epoch_eci_scores_2026-10-22.csv"
OUT = ROOT / "data/derived"
PANEL = OUT / "price_eci_panel.csv"   # снимок сопоставления, лежит в репозитории
INPUT_SHARE = 0.75          # смесь токенов вход:выход = 3:1
V_FRONT = 1.43              # ECI/мес, закрытый фронтир с 2025-07


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load():
    if not OR_JSON.exists():
        # сырой прайс-лист OpenRouter в репозиторий не кладём; панель — снимок на 22.10.2026
        return pd.read_csv(PANEL, parse_dates=["date"])
    raw = json.loads(OR_JSON.read_text(encoding="utf-8"))["data"]
    o = pd.DataFrame([dict(
        id=m["id"],
        p_in=float(m["pricing"]["prompt"] or 0) * 1e6,
        p_out=float(m["pricing"]["completion"] or 0) * 1e6,
    ) for m in raw])
    o = o[(o.p_in > 0) & ~o.id.str.contains(":free")]
    o["n"] = o.id.str.split("/").str[1].map(norm)
    e = pd.read_csv(ECI, parse_dates=["date"])
    e = e[e["Accessibility group"].isin(["Open weights", "Closed weights"])]
    e["n"] = e.Model.map(norm)
    m = e.merge(o, on="n", how="inner").drop_duplicates("Model")
    m["price"] = INPUT_SHARE * m.p_in + (1 - INPUT_SHARE) * m.p_out
    m["lp"] = np.log(m.price)
    m["open"] = m["Accessibility group"].eq("Open weights").astype(int)
    m["vintage"] = m.date.dt.year + (m.date.dt.dayofyear - 1) / 365.25
    m["org"] = m.Organization
    return m


def lower_envelope(m, grid):
    """Самая дешёвая цена достижения уровня ECI ≥ x (как у Epoch)."""
    rows = []
    for x in grid:
        for g, sub in m[m.eci >= x].groupby("open"):
            rows.append(dict(eci=x, open=g, min_price=sub.price.min()))
    return pd.DataFrame(rows).pivot(index="eci", columns="open", values="min_price")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = load()
    m.to_csv(OUT / "price_eci_panel.csv", index=False)
    print(f"Сопоставлено моделей: {len(m)} (открытых {m.open.sum()})")
    recent = m[m.date >= "2025-07-01"]
    specs = {
        "все, с 2025-07": ("lp ~ eci + open", recent),
        "+ поколение": ("lp ~ eci + open + vintage", recent),
        "+ лаб. FE": ("lp ~ eci + open + C(org)", recent),
        "только закрытые": ("lp ~ eci", recent[recent.open == 0]),
    }
    res = {}
    for k, (f, d) in specs.items():
        r = smf.ols(f, d).fit(cov_type="HC1")
        res[k] = r
        b, se = r.params["eci"], r.bse["eci"]
        g = r.params.get("open", np.nan)
        lam = 1 / (b * V_FRONT) if b > 0 else np.inf
        print(f"{k:16s} n={int(r.nobs):3d}  b={b:.3f}±{se:.3f}  "
              f"open γ={g:+.2f} (×{np.exp(g):.2f})  λ={lam:.1f} мес  R²={r.rsquared:.2f}")
    qr = smf.quantreg("lp ~ eci + open", recent).fit(q=0.5)
    print(f"медианная регр.  b={qr.params['eci']:.3f}  γ={qr.params['open']:+.2f} "
          f"(×{np.exp(qr.params['open']):.2f})")

    env = lower_envelope(recent, np.arange(140, 166, 2.0))
    env.columns = ["закрытые", "открытые"]
    env["закр/откр"] = env["закрытые"] / env["открытые"]
    print("\nНижняя огибающая: самая дешёвая цена за уровень ECI, $/млн токенов")
    print(env.round(3).to_string())
