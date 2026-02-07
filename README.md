# PortfolioLab

Portfolio optimization and financial analysis toolkit. Built with Python + Streamlit.

## What it does

- **Portfolio Optimizer** — Mean-variance (Markowitz) optimization with efficient frontier, plus Equal Weight and Risk Parity strategies. Supports weight constraints, backtesting, and CSV export.
- **AI Quant Analyst** — Chat with an AI about your portfolio. It calls real compute functions (not just vibes) to answer questions about risk, strategy comparison, correlations, and backtests.
- **Forensic Analysis** — Upload financial statements and detect red flags (cash conversion, debt-to-equity, revenue growth anomalies, etc.) with LLM-powered explanations.

## Quick start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/LLM-Powered-Forensic-Analysis-of-Financial-Statements.git
cd LLM-Powered-Forensic-Analysis-of-Financial-Statements

# Create virtual environment
python -m venv vevn
vevn\Scripts\activate        # Windows
# source vevn/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key (needed for AI features)
# Create a .env file in the root directory:
echo OPENAI_API_KEY=your_groq_api_key_here > .env

# Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). The portfolio optimizer works without it — only the AI Quant Analyst and forensic LLM explanations need the key.

## Data sources

- **Sample data** — Built-in synthetic data, works out of the box
- **Live market data** — Real prices from Yahoo Finance. Just type in any ticker symbols

## Project structure

```
app.py                          # Entry point
config.py                       # Settings and feature flags
src/
  core/                         # Types, interfaces, feature registry
  data/                         # Data providers (dummy, CSV, yfinance)
  features/
    portfolio/                  # Optimizer (Markowitz, risk parity, backtest)
    quant_analyst/              # AI chat with LLM tool-use
    forensic/                   # Financial red flag detection
  llm/                          # Groq/LLM client and prompt formatting
  ui/                           # Shared layout and components
```

Each feature is a self-contained module with its own service layer and UI. You can enable/disable any feature in `config.py`.

## Tech stack

- Python, Streamlit, Plotly
- scipy (optimization), scikit-learn, numpy, pandas
- yfinance (market data)
- Groq API with Llama 3.3 (LLM features)

## Adding features

The codebase uses a feature registry pattern. To add a new feature:

1. Create a new folder in `src/features/your_feature/`
2. Add `service.py` (inherits `BaseService`) and `page.py` (Streamlit renderer)
3. Register it in `app.py`'s `build_registry()` and add a flag in `config.py`

That's it. The sidebar and routing handle themselves.
