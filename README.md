# PortfolioLab

PortfolioLab is a work‑in‑progress playground for portfolio optimization, basic quant experiments, and some AI‑assisted financial analysis. Built with Python + Streamlit.

## What it does

- **Portfolio optimizer**
  - Mean–variance (Markowitz) optimization
  - Equal‑weight and simple risk‑parity style allocations
  - Efficient frontier plots, weight charts, and basic backtests

- **AI quant helper**
  - Chat interface that can call real Python functions (risk, correlations, strategy comparisons, simple backtests)
  - Uses Groq + LLM models under the hood for the AI parts

- **Forensic-ish analysis**
  - Upload financial statements (CSV/Excel style)
  - Runs rule‑based and ML checks for:
    - Cash conversion issues
    - Leverage / debt ratios
    - Growth anomalies
  - LLM explains the red flags in plain language

A lot of this is still rough and subject to change.

## Project status

- **Status:** Work in progress / prototype  
- **Good for:** experimenting with ideas, hacking on features, trying LLM + tools patterns  
- **Not ready for:** production, real-money decision making, or stable public APIs


## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/MohitWarrier/LLM-Powered-Forensic-Analysis-of-Financial-Statements.git
cd LLM-Powered-Forensic-Analysis-of-Financial-Statements
```

### 2. Create and Activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Groq API Key
The portfolio optimizer works without any API keys.
The AI quant helper and forensic explanations need an LLM.

Create a .env file in the project root:
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```
Make sure the variable name matches what the code expects (for example GROQ_API_KEY or OPENAI_API_KEY if you wired it that way).

### 5. Run the App
```bash
streamlit run app.py
```

