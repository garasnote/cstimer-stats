# 🧩 cstimer Stats Dashboard

> **[👉 Try it live on Streamlit](https://cstimer-stats.streamlit.app/)**  
> No install, no Python, just upload your file and go.

A statistics dashboard for [cstimer](https://cstimer.net) exports. Upload your JSON solve history and get a full breakdown of your performance — rolling averages, PB tracking, distribution fitting, confidence intervals, and a scramble visualizer.

> ⚠️ Mostly written by [Claude](https://claude.ai) (Anthropic). Proudly AI slop.

---

## Features

- **📊 Overview** — solve count, best, mean, std dev, and a PB motivator that tells you what time you have a 5% chance of hitting on your next solve
- **📈 Time Series** — scatter of all solves with Ao5/Ao12/Ao100 rolling averages, PB progression, and cumulative 95% confidence intervals
- **📉 Distribution** — fits Log-normal, Ex-Gaussian, Gamma, and Normal to your data, ranks them by AIC/BIC, and lets you pick whichever you want (Normal included, not recommended — it allows negative solve times, which would be impressive)
- **🔀 Scramble Viewer** — applies any scramble to a 2D cube net so you can see the starting state
- **📋 Solve Log** — full table with +2 and DNF penalties highlighted

## How to use

1. Open [cstimer.net](https://cstimer.net)
2. Go to **Options → Export** — this downloads a small JSON file (usually under 100KB)
3. Go to **[cstimer-stats.streamlit.app](https://cstimer-stats.streamlit.app/)**
4. Upload the file using the panel on the left
5. Select which sessions to analyze

## Run locally

```bash
git clone https://github.com/garasnote/cstimer-stats.git
cd cstimer-stats
pip install -r requirements.txt
streamlit run app.py
```

## Distribution models

The app fits four distributions to your solve times and ranks them by [AIC/BIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) (lower = better fit, penalized for complexity):

| Model | Params | Notes |
|---|---|---|
| Log-normal | 2 | Usually wins. Always positive, right-skewed — matches human performance data well |
| Ex-Gaussian | 3 | Gold standard in psychometrics for reaction times. Captures the long tail of bad solves |
| Gamma | 2 | Always positive, flexible shape |
| Normal | 2 | Included for completeness. Can predict negative solve times. You've been warned |

The winning model is used for the cumulative 95% CI band and the PB predictor. You can override it from the sidebar dropdown.

## Stack

- [Streamlit](https://streamlit.io) — UI
- [Plotly](https://plotly.com/python/) — charts and animations
- [SciPy](https://scipy.org) — distribution fitting
- [Pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org) — data wrangling
