"""Лаг открытых моделей за закрытым фронтиром по Epoch Capabilities Index.

Лаг модели m = дата(m) − первая дата, когда закрытый фронтир достиг ECI(m).
Лаг месяца t = t − первая дата, когда закрытый фронтир достиг лучшего
открытого ECI на момент t (горизонтальный разрыв, как у Epoch).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ECI = ROOT / "data/epoch_eci_scores_2026-10-22.csv"
OUT = ROOT / "data/derived"
DAYS_PER_MONTH = 30.44


def load():
    e = pd.read_csv(ECI, parse_dates=["date"])
    e["open"] = e["Accessibility group"].eq("Open weights")
    e["closed"] = e["Accessibility group"].eq("Closed weights")
    return e.sort_values("date").reset_index(drop=True)


def frontier(e, mask):
    """Ступенчатый фронтир: (дата, running max ECI) по подвыборке."""
    f = e.loc[mask, ["date", "eci"]].sort_values("date").copy()
    f["front"] = f["eci"].cummax()
    return f.drop_duplicates("front")[["date", "front"]].reset_index(drop=True)


def first_reach(front, level):
    """Первая дата, когда фронтир достиг уровня (NaT, если не достиг)."""
    hit = front.loc[front["front"] >= level, "date"]
    return hit.iloc[0] if len(hit) else pd.NaT


def model_lags(e, cf):
    rows = []
    for _, r in e[e["open"]].iterrows():
        d0 = first_reach(cf, r["eci"])
        if pd.isna(d0):          # открытая модель выше закрытого фронтира
            lag = 0.0
        else:
            lag = max(0.0, (r["date"] - d0).days / DAYS_PER_MONTH)
        rows.append(dict(model=r["Model"], org=r["Organization"],
                         country=r["Country (of organization)"],
                         date=r["date"], eci=r["eci"], lag_m=lag))
    return pd.DataFrame(rows)


def monthly_gap(e, cf):
    of = frontier(e, e["open"])
    grid = pd.date_range("2024-01-01", e["date"].max(), freq="MS")
    rows = []
    for t in grid:
        o = of.loc[of["date"] <= t, "front"]
        c = cf.loc[cf["date"] <= t, "front"]
        if not len(o) or not len(c):
            continue
        d0 = first_reach(cf, o.iloc[-1])
        lag = (t - d0).days / DAYS_PER_MONTH if not pd.isna(d0) else 0.0
        rows.append(dict(month=t, open_front=o.iloc[-1], closed_front=c.iloc[-1],
                         eci_gap=c.iloc[-1] - o.iloc[-1], lag_m=lag))
    return pd.DataFrame(rows)


def eci_speed(cf, start="2024-01-01"):
    """Скорость фронтира, пунктов ECI в месяц (МНК по ступеням)."""
    f = cf[cf["date"] >= start]
    x = (f["date"] - f["date"].min()).dt.days / DAYS_PER_MONTH
    return np.polyfit(x, f["front"], 1)[0]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    e = load()
    cf = frontier(e, e["closed"])
    ml = model_lags(e, cf)
    mg = monthly_gap(e, cf)
    ml.to_csv(OUT / "open_model_lags.csv", index=False)
    mg.to_csv(OUT / "monthly_gap.csv", index=False)

    print(f"Скорость закрытого фронтира: {eci_speed(cf):.2f} ECI/мес с 2024; "
          f"{eci_speed(cf, '2025-07-01'):.2f} с 2025-07")
    print("\nГоризонтальный разрыв открытых моделей (мес), по кварталам:")
    q = mg.set_index("month")["lag_m"].resample("QS").mean()
    print(q.round(1).to_string())

    # «лучшая открытая модель на дату» — лаг лучших, а не всех
    best = ml[ml["eci"] >= ml.groupby(ml["date"].dt.to_period("Q"))["eci"].transform("max") - 1.5]
    print("\nЛаг лучших открытых моделей квартала (±1.5 ECI от максимума):")
    print(best.groupby(best["date"].dt.to_period("Q"))["lag_m"].agg(["mean", "count"]).round(1).to_string())

    # естественный эксперимент: цели отчёта о дистилляции 23.02.2026
    tgt = {"DeepSeek", "Moonshot", "MiniMax"}
    ml["target"] = ml["org"].isin(tgt)
    ml["post"] = ml["date"] >= "2026-03-01"
    sub = ml[ml["date"] >= "2025-03-01"]
    print("\nЛаг моделей (мес): цели отчёта vs прочие открытые, до/после марта 2026")
    print(sub.groupby(["target", "post"])["lag_m"].agg(["mean", "median", "count"]).round(1).to_string())
    print("\nМодели целевых лабораторий с 2025-03:")
    print(sub[sub["target"]].sort_values("date")[["model", "date", "eci", "lag_m"]].round(1).to_string(index=False))
