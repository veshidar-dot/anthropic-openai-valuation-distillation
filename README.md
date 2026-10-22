# Valuing Anthropic and OpenAI Under Model Distillation

**The lead over open-weight models implied by AI lab valuations, 2024–2026**

Danila Krasnov, European University at St. Petersburg (d.krasnov@eu.spb.ru).
Working paper, version of 22 October 2026: [paper/main.pdf](paper/main.pdf).

## Question

How many months must Anthropic stay ahead of open-weight models such as DeepSeek, Kimi (Moonshot AI), GLM (Zhipu) and MiniMax to justify a $2 trillion IPO valuation, and how much is each month of lead worth when competitors can distil the frontier models' outputs?

## Abstract

Valuation practice has long backed out of stock prices a firm's competitive advantage period, the time over which it is expected to earn excess returns, but that period is normally unobservable. For frontier AI labs it has a measurable counterpart: the lead over the best open-weight model on the Epoch Capabilities Index (ECI), which was about 1 month for Anthropic through 2025 and 7.7 months in October 2026. Regressing OpenRouter API prices on ECI for models released since July 2025 shows that the price premium for capability decays with a time constant of 6.9 months (90% interval 5.2–11.0); closed weights sell at 2.6 times the price of equally capable open weights, a smaller gap than the tenfold one reported by Demirer, Fradkin, Tadelis and Peng. A discounted cash flow (DCF) model built on this margin curve is inverted for 20 valuation events of Anthropic, OpenAI, xAI, Mistral, Cohere, Zhipu, MiniMax and Moonshot. The main result is a set of iso-value curves of revenue growth against implied lead: valuations do not identify the lead without a growth forecast. With the revenue path reported for Anthropic's IPO ($190–200 billion in 2028), a $2 trillion valuation requires a lead of 7–10 months or more when the price elasticity of demand is 0.5, and is out of reach at almost any lead when the elasticity exceeds one; OpenAI's $852 billion round of March 2026 implied its 8-month lead would shrink to 0–5 months. One month of lead is worth 2–13% of a frontier lab's value, depending mainly on the demand elasticity. Labs without a lead are priced on revenue growth of roughly 200–400% or on gross margins two to four times those they report. Anthropic's February 2026 countermeasures against distillation did not change the lag of the targeted labs relative to other open-weight developers.

## Key numbers

| Quantity | Value |
|---|---|
| Anthropic's lead over the best open-weight model, Oct 2026 | 7.7 months |
| Same, median Jan 2025 – Oct 2026 | 1.5 months |
| Decay time constant of the capability price premium, λ | 6.9 months (5.2–11.0) |
| Price of closed vs equally capable open weights | ×2.6 |
| Value of one month of lead | 2–13% of firm value |
| Anthropic DCF at its current lead (12% discount rate, bankers' growth path) | $1.36 trillion |
| Same, if the lead reverts to its 2025–26 median | $331 billion |
| Lead implied by OpenAI's $852bn round (Mar 2026), observed 8.0 | 0–5 months |
| Difference-in-differences, lag of DeepSeek/Moonshot/MiniMax after the Feb 2026 distillation report | +0.1 months (s.e. 1.7) |

## Repository

```
data/epoch_eci_scores_2026-10-22.csv   Epoch Capabilities Index snapshot (CC BY 4.0, Epoch AI)
data/inputs/events.csv                 valuation panel: 20 events, sources and confidence grades
data/derived/price_eci_panel.csv       107 models matched to OpenRouter prices, 22 Oct 2026
data/derived/*.csv                     lag series, open-model lags, iso-value curves
src/lag_series.py                      lead and lag measures (Section 3)
src/price_premium.py                   hedonic regressions (Section 4)
src/dcf.py                             DCF model with lead-dependent margin (Section 5)
src/implied.py                         implied lead and implied margin solvers
src/paper_outputs.py                   all figures and tables of the paper
paper/                                 LaTeX source, figures, tables, PDF
```

Code comments are in Russian; function and variable names are in English.

## Reproducing the paper

```bash
pip install -r requirements.txt
python -P src/paper_outputs.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

`paper_outputs.py` rewrites `paper/figures/` and `paper/tables/` from the files in `data/`. The raw OpenRouter price list is not redistributed; the matched price panel in `data/derived/price_eci_panel.csv` is the snapshot used in the paper. To rebuild it from a fresh download, save `https://openrouter.ai/api/v1/models` as `data/raw/openrouter/models_2026-10-22.json` and run `python -P src/price_premium.py` (a new download will reflect current prices, not those of 22 October 2026).

## Data caveats

Several valuation inputs come from press reports (Bloomberg, Reuters, FT, WSJ, CNBC) rather than filings, and some run-rates are interpolated; each row of `data/inputs/events.csv` carries its source and a confidence grade. Anthropic's gross margin has not been disclosed with a fixed definition. xAI's figures are SpaceX's AI segment, which includes advertising on X and compute rental.

## Disclosure

Data collection, code and drafting were assisted by Claude, a large language model developed by Anthropic, one of the firms studied. The author has no financial or professional relationship with any of the firms studied.

## Licence

Code: MIT. Compiled data: CC BY 4.0. Epoch AI data: CC BY 4.0, credit Epoch AI. See [LICENSE](LICENSE).

## Citation

Krasnov, D. (2026). Valuing Anthropic and OpenAI Under Model Distillation: The Lead over Open-Weight Models Implied by AI Lab Valuations, 2024–2026. Working paper, European University at St. Petersburg.
