# Condensed Mathematics for ETFs

Uses Clausen-Scholze condensed sets to handle the "large" topological spaces arising from continuous-time limit order book data. Provides a rigorous framework for mixing discrete (tick-level) and continuous (diffusion) phenomena. The per‑ETF score is the condensed cohomology rank.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Discrete-continuous interface via quantised and smoothed returns
- Condensed profunctor (sheaf-like structure)
- Score = cohomology rank (higher = more structure)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-condensed-mathematics-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High cohomology rank → more structure at the discrete-continuous interface.
- Low cohomology rank → simpler interface.

## Requirements

See `requirements.txt`.
