# Do AI Lab Valuations Price the Frontier Lead?

**Measured and implied leads of Anthropic and OpenAI over open-weight models, 2024–2026**

Danila Krasnov, European University at St. Petersburg (d.krasnov@eu.spb.ru).
Working paper, revised version of 23 October 2026: [paper/main.pdf](paper/main.pdf).

## Question

How many months must Anthropic stay ahead of open-weight models such as DeepSeek, Kimi (Moonshot AI), GLM (Zhipu) and MiniMax to justify a $2 trillion IPO valuation?

## Abstract

Valuation practice backs a competitive advantage period out of stock prices, but that period is normally unobservable. For frontier AI labs it has a measurable counterpart: the lead over the best open-weight model on the Epoch Capabilities Index (ECI). Anthropic's lead was about 1 month through 2025 and 7.7 months in October 2026 (90% interval 5.8–11.6). Whether the lead earns a price premium is not identified by a cross-section of API prices: on OpenRouter, price rises with capability at the same rate for open-weight models, which sell near serving cost, as for closed ones, so the rent component of the slope is 0.013 log points per ECI point with a 90% interval of −0.065 to 0.094. I therefore treat the whole slope as an upper bound on the lead premium, which then decays with a time constant of about 7 months, and invert a discounted cash flow (DCF) model for 20 valuation events of Anthropic, OpenAI, xAI, Mistral, Cohere, Zhipu, MiniMax and Moonshot. The main result is a set of iso-value curves of revenue growth against implied lead: valuations do not identify the lead without a growth forecast. Two conclusions survive a sensitivity analysis over the premium's decay, the demand elasticity, the margin calibration and cap, the growth forecast and reinvestment. With the revenue path reported for Anthropic's IPO ($190–200 billion in 2028), a $2 trillion valuation cannot be reached at any lead in 13 of 14 specifications, so it presumes faster growth. OpenAI's $852 billion round of March 2026 implied a lead of 0–5 months, below the lower bound of its measured 8.0 months. Even at the upper bound of the premium, a month of lead is worth 0–10% of a frontier lab's value when demand elasticity is at least one, about 5% in the baseline. Labs without a lead are priced on revenue growth of roughly 200–400%. Anthropic's February 2026 countermeasures against distillation had no detectable effect on the lag of the targeted labs, but the test cannot rule out effects of up to three months.

## Key numbers

| Quantity | Value |
|---|---|
| Anthropic's lead over the best open-weight model, Oct 2026 | 7.7 months (90% interval 5.8–11.6) |
| Same, median Jan 2025 – Oct 2026 | 1.5 months |
| Price slope in capability, all models | 0.101 log points per ECI point |
| Rent component (closed minus open slope) | 0.013 (90% interval −0.065 to 0.094): not identified |
| Decay time constant of the lead premium, upper bound | 6.9 months (3.6–11.0 across specifications) |
| Price of closed vs equally capable open weights | ×2.6 |
| Value of one month of lead | about 5% baseline; 0–10% with demand elasticity ≥ 1; 13% at 0.5 |
| Anthropic DCF at its current lead (12%, bankers' growth path) | $1.36 trillion ($0.92–1.92 trillion across specifications) |
| $2 trillion attainable at some lead on the bankers' growth path | in 1 of 14 specifications |
| Lead implied by OpenAI's $852bn round (Mar 2026); measured 8.0 (6.6–10.5) | 0–5.4 months in every specification |
| Difference-in-differences, lag of DeepSeek/Moonshot/MiniMax after the Feb 2026 distillation report | +0.1 months (s.e. 1.7) |

The first version (22 October 2026) used a demand elasticity of 0.5 and read the whole price slope as rent; a review pointed out that neither was defended. The revision makes 1.1 the baseline, separates cost from rent in the slope, adds intervals for the lead and a one-at-a-time sensitivity table, and narrows the claims to what survives them.

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
src/paper_outputs.py                   figures and main tables of the paper
src/robustness.py                      cost/rent decomposition, intervals for the lead, sensitivity table
paper/                                 LaTeX source, figures, tables, PDF
```

Code comments are in Russian; function and variable names are in English.

## Reproducing the paper

```bash
pip install -r requirements.txt
python -P src/paper_outputs.py
python -P src/robustness.py
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

Krasnov, D. (2026). Do AI Lab Valuations Price the Frontier Lead? Measured and Implied Leads of Anthropic and OpenAI over Open-Weight Models, 2024–2026. Working paper, European University at St. Petersburg.
