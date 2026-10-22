"""Таблицы и рисунки препринта: paper/figures/*.pdf, paper/tables/*.tex.

Всё пересчитывается из data/ и src/ одним запуском:
  python -P src/paper_outputs.py
"""
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcf  # noqa: E402
import implied  # noqa: E402
import lag_series  # noqa: E402
import price_premium  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
TAB = ROOT / "paper/tables"
G_GRID = np.round(np.arange(0.40, 3.0001, 0.05), 2)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150,
})
GREYS = ["0.0", "0.35", "0.55", "0.75"]
STYLES = ["-", "--", "-.", ":"]


def events():
    return pd.read_csv(implied.EVENTS, parse_dates=["date"])


def eci():
    return pd.read_csv(implied.ECI, parse_dates=["date"])


def row(firm, d):
    ev = events()
    return ev[(ev.firm == firm) & (ev.date == d)].iloc[0]


OPEN_LABS = {"Zhipu", "MiniMax", "Moonshot"}
GM_REF = 0.75               # GM закрытой лаборатории с отрывом L_REF
L_REF = 6.0


def l_obs(r):
    v = implied.lead_months(eci(), r.eci_org, r.date)
    return v if v == v else 0.0


def firm_at(r, g0, wacc=0.12, gm_ref=None):
    """Одна отраслевая кривая GM(L) для всех; фирма стоит на ней в точке L_obs."""
    k = dcf.industry_kappa(GM_REF if gm_ref is None else gm_ref, L_REF)
    return dcf.Firm(r.firm, rev0=r.rr_usd_b, g0=g0, gm_now=r.gm_now,
                    gm_target=float("nan"), l_now=l_obs(r), cash=r.cash_usd_b,
                    tau=2.5, wacc=wacc, kappa=k, open_w=r.firm in OPEN_LABS)


def iso_lag(r, wacc=0.12):
    return [implied.implied_lag(firm_at(r, g, wacc), r.value_usd_b) for g in G_GRID]


def implied_g0(r, L, wacc=0.12):
    """Рост, при котором цена оправдана, если отрыв останется равным L."""
    lo, hi = 0.0, 6.0
    f = lambda g: dcf.value(firm_at(r, g, wacc), L)["equity"]
    if f(hi) < r.value_usd_b:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < r.value_usd_b else (lo, mid)
    return (lo + hi) / 2


# ---------- рисунок 1: лидерство и разрыв ----------
def fig_leads():
    e = lag_series.load()
    months = pd.date_range("2024-10-01", "2026-10-01", freq="MS")
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for i, org in enumerate(["Anthropic", "OpenAI"]):
        ax.plot(months, [implied.lead_months(e, org, t) for t in months],
                color=GREYS[i], ls=STYLES[i], lw=1.4, label=f"{org}: lead over best open-weight model")
    cf = lag_series.frontier(e, e["closed"])
    mg = lag_series.monthly_gap(e, cf)
    mg = mg[mg.month >= months[0]]
    ax.plot(mg.month, mg.lag_m, color=GREYS[2], ls=":", lw=1.4,
            label="Horizontal gap, closed frontier vs best open-weight")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(pd.Timestamp("2026-02-23"), color="0.6", lw=0.8)
    ax.text(pd.Timestamp("2026-02-28"), -3.6, "distillation\nreport", fontsize=8, color="0.4")
    ax.set_ylabel("Months")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_leads.pdf")
    plt.close(fig)


# ---------- рисунок 2: гедоническая регрессия ----------
def fig_hedonic(m):
    r = m[m.date >= "2025-07-01"]
    fit = smf.ols("lp ~ eci + open", r).fit()
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    for g, mk, lab in [(0, "o", "Closed weights"), (1, "^", "Open weights")]:
        s = r[r.open == g]
        ax.scatter(s.eci, s.price, marker=mk, s=16, facecolors="none" if g else "0.2",
                   edgecolors="0.2", lw=0.7, label=lab)
        x = np.linspace(s.eci.min(), s.eci.max(), 50)
        ax.plot(x, np.exp(fit.params.Intercept + fit.params.eci * x + fit.params.open * g),
                color="0.2", ls="--" if g else "-", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch Capabilities Index")
    ax.set_ylabel("Blended price, USD per million tokens")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_hedonic.pdf")
    plt.close(fig)


# ---------- рисунок 3: изо-кривые (главный результат) ----------
ISO_EVENTS = {
    "Anthropic": [("2025-09-02", "Series F, Sep 2025"), ("2026-02-12", "Series G, Feb 2026"),
                  ("2026-05-28", "Series H, May 2026"), ("2026-10-15", "IPO talk, Oct 2026")],
    "OpenAI": [("2024-10-02", "Round, Oct 2024"), ("2025-10-15", "Secondary, Oct 2025"),
               ("2026-03-31", "Round, Mar 2026")],
}


def fig_iso():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), sharey=True)
    rows = []
    for ax, (firm, evs) in zip(axes, ISO_EVENTS.items()):
        for i, (d, lab) in enumerate(evs):
            r = row(firm, d)
            ys = iso_lag(r)
            ys = [np.nan if y is None else y for y in ys]
            ax.plot(G_GRID * 100, ys, color=GREYS[i], ls=STYLES[i], lw=1.3, label=lab)
            L0 = l_obs(r)
            g_star = implied_g0(r, L0)
            if g_star == g_star:
                ax.plot(g_star * 100, L0, marker="o", color=GREYS[i], ms=4)
            for g, y in zip(G_GRID, ys):
                rows.append(dict(firm=firm, event=lab, g0=g, L=y))
        for (f_, d), (mult, yrs, _) in implied.ANCHORS.items():
            if f_ != firm:
                continue
            g_a = implied.g0_for_path(mult, yrs)
            ax.axvline(g_a * 100, color="0.5", lw=0.8, ls=(0, (1, 2)))
            ax.text(g_a * 100 + 3, 15.3, "forecast\nanchor", fontsize=7, color="0.4")
        ax.set_title(firm, fontsize=10)
        ax.set_xlabel("First-year revenue growth $g_0$, %")
        ax.set_xlim(40, 300)
        ax.set_ylim(0, 17)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Implied lead $L$, months")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_iso.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT / "data/derived/iso_curves.csv", index=False)


# ---------- рисунок 4: подразумеваемая маржа без отрыва ----------
GM_EVENTS = [("Zhipu", "2026-10-16", "Zhipu, Oct 2026"), ("MiniMax", "2026-10-16", "MiniMax, Oct 2026"),
             ("Moonshot", "2026-10-06", "Moonshot, Oct 2026"), ("Mistral", "2026-09-08", "Mistral, Sep 2026")]


def fig_gm():
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for i, (firm, d, lab) in enumerate(GM_EVENTS):
        r = row(firm, d)
        ys = [implied.implied_gm(r, g, 2.5, 0.12) * 100 for g in G_GRID]
        ax.plot(G_GRID * 100, ys, color=GREYS[i], ls=STYLES[i], lw=1.3, label=lab)
        if firm in ("Zhipu", "MiniMax"):
            ax.plot(305, r.gm_now * 100, marker="<", color=GREYS[i], ms=5, clip_on=False)
    ax.text(297, 29, "reported H1 2026 gross margin", fontsize=7, color="0.4", ha="right")
    ax.set_xlim(40, 300)
    ax.set_ylim(0, 90)
    ax.set_xlabel("First-year revenue growth $g_0$, %")
    ax.set_ylabel("Implied long-run gross margin, %")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_gm.pdf")
    plt.close(fig)


# ---------- таблицы ----------
def tex_table(path, body):
    path.write_text(body, encoding="utf-8")


def tab_hedonic(m):
    r = m[m.date >= "2025-07-01"]
    specs = [("Baseline", "lp ~ eci + open", r),
             ("Vintage control", "lp ~ eci + open + vintage", r),
             ("Developer fixed effects", "lp ~ eci + open + C(org)", r),
             ("Closed weights only", "lp ~ eci", r[r.open == 0])]
    lines = []
    for name, f, d in specs:
        q = smf.ols(f, d).fit(cov_type="HC1")
        b, se = q.params["eci"], q.bse["eci"]
        g = q.params.get("open", np.nan)
        gse = q.bse.get("open", np.nan)
        lam = 1 / (b * price_premium.V_FRONT)
        gtxt = "---" if g != g else f"{g:.2f} ({gse:.2f})"
        etxt = "---" if g != g else f"{math.exp(g):.2f}"
        lines.append(f"{name} & {b:.3f} ({se:.3f}) & {gtxt} & {etxt} & {lam:.1f} & {int(q.nobs)} & {q.rsquared:.2f} \\\\")
    tex_table(TAB / "hedonic.tex", "\n".join(lines) + "\n")


def tab_events():
    e = eci()
    lines = []
    for r in events().itertuples():
        L0 = implied.lead_months(e, r.eci_org, r.date)
        g_s = implied_g0(r, L0 if L0 == L0 else 0.0)
        Ltxt = "n/a" if L0 != L0 else f"${L0:.1f}$"
        gtxt = "$>$600" if g_s != g_s else f"{g_s * 100:.0f}"
        lines.append(f"{r.firm} & {r.date:%Y-%m} & {r.event} & {r.value_usd_b:,.1f} & "
                     f"{r.rr_usd_b:.2f} & {r.value_usd_b / r.rr_usd_b:.0f} & {Ltxt} & {gtxt} & {r.conf} \\\\")
    tex_table(TAB / "events.tex", "\n".join(lines) + "\n")


def anchored_one(firm, d, wacc, gm_ref):
    mult, yrs, _ = implied.ANCHORS[(firm, d)]
    r = row(firm, d)
    f = firm_at(r, implied.g0_for_path(mult, yrs), wacc, gm_ref)
    L0 = l_obs(r)
    v0 = dcf.value(f, L0)["equity"]
    dv = (dcf.value(f, L0 + 1)["equity"] - dcf.value(f, L0 - 1)["equity"]) / 2
    return v0, implied.implied_lag(f, r.value_usd_b), dv / v0


def tab_anchored():
    """Две панели: эластичность спроса 0,5 (база) и 1,1 (Demirer et al.)."""
    lines = []
    for eps in (0.5, 1.1):
        dcf.EPS = eps
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Price elasticity of demand {eps}}}}} \\\\")
        for wacc, gm, cap in [(0.12, 0.75, 0.85), (0.10, 0.75, 0.85), (0.12, 0.85, 0.92), (0.10, 0.85, 0.92)]:
            dcf.GM_CAP = cap
            cells = []
            for firm, d in implied.ANCHORS:
                v0, L, el = anchored_one(firm, d, wacc, gm)
                cells += [f"{v0 / 1000:.2f}", "unattainable" if L is None else f"{L:.1f}", f"{el * 100:.0f}"]
            lines.append(f"{wacc * 100:.0f}\\% & {gm * 100:.0f}\\% & " + " & ".join(cells) + " \\\\")
        if eps == 0.5:
            lines.append("\\addlinespace")
    dcf.GM_CAP, dcf.EPS = 0.85, 0.5
    tex_table(TAB / "anchored.tex", "\n".join(lines) + "\n")


def tab_did():
    e = lag_series.load()
    cf = lag_series.frontier(e, e["closed"])
    ml = lag_series.model_lags(e, cf)
    ml = ml[ml.date >= "2025-03-01"].copy()
    ml["target"] = ml.org.isin({"DeepSeek", "Moonshot", "MiniMax"}).astype(int)
    ml["post"] = (ml.date >= "2026-03-01").astype(int)
    q = smf.ols("lag_m ~ target * post", ml).fit(cov_type="HC1")
    g = ml.groupby(["target", "post"]).lag_m.agg(["mean", "count"])
    lines = []
    for t, name in [(1, "Named labs (DeepSeek, Moonshot, MiniMax)"), (0, "Other open-weight developers")]:
        pre, post = g.loc[(t, 0)], g.loc[(t, 1)]
        lines.append(f"{name} & {pre['mean']:.1f} ({int(pre['count'])}) & {post['mean']:.1f} ({int(post['count'])}) & {post['mean'] - pre['mean']:+.1f} \\\\")
    b, se = q.params["target:post"], q.bse["target:post"]
    lines.append(f"\\midrule\nDifference-in-differences & & & {b:+.1f} ({se:.1f}) \\\\")
    tex_table(TAB / "did.tex", "\n".join(lines) + "\n")
    return b, se


def tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"$\sim$"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def tab_sources():
    lines = []
    for r in events().itertuples():
        url, _, note = str(r.source).partition(" ")
        if "." not in url.split("/")[0]:      # источник без адреса
            url, note = "", str(r.source)
        u = f"\\url{{{url}}}" if url else ""
        n = f" {tex_escape(note)}" if note else ""
        lines.append(f"{r.firm} & {r.date:%Y-%m-%d} & {tex_escape(r.event)} & {u}{n} \\\\")
    tex_table(TAB / "sources.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    m = price_premium.load()
    fig_leads()
    fig_hedonic(m)
    fig_iso()
    fig_gm()
    tab_hedonic(m)
    tab_events()
    tab_anchored()
    tab_sources()
    b, se = tab_did()
    print(f"DiD {b:+.2f} ({se:.2f})")
    for p in sorted(TAB.glob("*.tex")):
        print(f"--- {p.name}\n{p.read_text(encoding='utf-8')}")
