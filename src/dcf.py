"""DCF лаборатории с маржой, зависящей от лага имитации L.

Маржа (вывод в notes.md):
  цена фронтира p_F = p_open · e^{-γ} · e^{L/λ}   (closedness × лидерство)
  себестоимость   c_F = κ · p_open
  GM(L) = 1 − κ·e^{γ}·e^{−L/λ}
κ калибруется так, чтобы GM(L_now) = gm_target — долгосрочной марже,
которую компания планирует при нынешнем лаге (текущая gm_now занижена
субсидиями и дефицитом мощностей и задаёт только переход). γ, λ — из
price_premium.py.

Выручка при той же себестоимости токена масштабируется как
((1−gm_target)/(1−GM))^(1−ε): дороже токен — меньше объём (ε=0.5).
Если L мал, компания может бросить гонку и стать хостингом своих моделей:
GM 25 % (как у открытых хостеров), R&D 6 %, доля выручки −20 %. Берётся
лучшая из двух стратегий; капитал не ниже ликвидности (ограниченная
ответственность).
"""
import math
from dataclasses import dataclass, replace

EXP_GAMMA = 0.38            # e^γ, бутстрэп [0.22; 0.61]
LAM = 6.9                   # мес, бутстрэп [5.2; 11.0]
EPS = 1.1                   # эластичность спроса; Demirer et al. — чуть выше 1 (было 0.5 до рецензии)
GM_CAP = 0.85
TG = 0.03
TAX = 0.21
SGA = 0.12
RD_SS = 0.22
REINV = 0.25                # доля прироста выручки в ЦОДы и оборотку
ROIC_TERM = 0.25
YEARS = 10


@dataclass(frozen=True)
class Firm:
    name: str
    rev0: float             # выручка ближайших 12 мес на дату оценки, $ млрд
    g0: float               # рост выручки в 1-й год прогноза
    gm_now: float
    gm_target: float        # долгосрочная GM при нынешнем лаге
    l_now: float            # мес лидерства над лучшей открытой моделью
    cash: float = 0.0
    rd0: float = 0.45
    tau: float = 2.5        # лет, затухание роста к TG
    wacc: float = 0.12
    kappa: float | None = None   # общий для отрасли κ вместо калибровки по фирме
    open_w: bool = False         # открытые веса: премии закрытости нет


GM_COMMODITY = 0.25


def industry_kappa(gm_ref=0.75, l_ref=6.0):
    """κ, при котором закрытая лаборатория с отрывом l_ref имеет GM gm_ref."""
    return (1 - gm_ref) / (EXP_GAMMA * math.exp(-l_ref / LAM))


def kappa(f):
    if f.kappa is not None:
        return f.kappa
    return (1 - f.gm_target) / (EXP_GAMMA * math.exp(-max(f.l_now, 0.0) / LAM))


def gm_of(f, L):
    eg = 1.0 if f.open_w else EXP_GAMMA
    return min(GM_CAP, 1 - kappa(f) * eg * math.exp(-L / LAM))


def _value(f, L, chase):
    gm_ss = gm_of(f, L) if chase else GM_COMMODITY
    gm_ref = max(gm_of(f, f.l_now), GM_COMMODITY)   # цена на дату оценки
    price_f = ((1 - gm_ref) / (1 - max(gm_ss, -1.0))) ** (1 - EPS)
    if not chase:
        price_f *= 0.8
    rev, pv, disc, nopat = f.rev0, 0.0, 1.0, 0.0
    for i in range(YEARS):
        g = TG + (f.g0 - TG) * math.exp(-i / f.tau)
        rev_prev = rev
        rev = rev * (1 + g)
        w = min(1.0, (i + 1) / 3)                 # маржа сходится за 3 года
        gm_y = f.gm_now + (gm_ss - f.gm_now) * w
        rev_y = rev * (1 + (price_f - 1) * w)
        rd = (RD_SS + (f.rd0 - RD_SS) * math.exp(-i / 2.5)) if chase else 0.06
        ebit = rev_y * (gm_y - rd - SGA)
        nopat = ebit * (1 - TAX) if ebit > 0 else ebit
        fcf = nopat - REINV * (rev - rev_prev) * (1 + (price_f - 1) * w)
        disc /= 1 + f.wacc
        pv += fcf * disc
    tv = max(nopat, 0) * (1 + TG) * (1 - TG / ROIC_TERM) / (f.wacc - TG)
    equity = max(pv + tv * disc, 0.0) + f.cash
    return dict(equity=equity, pv_fcf=pv, pv_tv=tv * disc,
                gm_ss=gm_ss, chase=chase)


def value(f, L):
    a, b = _value(f, L, True), _value(f, L, False)
    return a if a["equity"] >= b["equity"] else b


def implied_lag(f, target, lo=0.0, hi=60.0):
    """L, при котором DCF = target; None, если вне [lo, hi]."""
    if value(f, hi)["equity"] < target:
        return None
    if value(f, lo)["equity"] > target:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if value(f, mid)["equity"] < target else (lo, mid)
    return (lo + hi) / 2


def with_(f, **kw):
    return replace(f, **kw)
