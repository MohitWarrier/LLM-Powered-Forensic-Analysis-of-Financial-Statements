"""
LLM-callable tool definitions for the AI Quant Analyst.

Each tool is a thin wrapper around existing optimizer/data functions.
TOOL_DEFINITIONS contains the JSON schemas for OpenAI function calling.
TOOL_REGISTRY maps function names to callables.

All tool functions receive a `context` dict with:
    - prices: pd.DataFrame (loaded price data)
    - weights: dict[str, float] (current portfolio weights)
    - optimizer: MeanVarianceOptimizer instance
    - risk_free_rate: float
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Tool Implementations ─────────────────────────────────────────

def get_portfolio_metrics(context: dict, **kwargs) -> dict:
    """Compute portfolio risk/return metrics using current weights."""
    prices = context["prices"]
    weights = context["weights"]
    optimizer = context["optimizer"]
    rf = context["risk_free_rate"]

    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    returns = prices[tickers].pct_change().dropna()

    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252

    port_return = float(np.dot(w, mean_ret.values))
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov.values, w))))
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

    # Risk metrics
    port_returns = returns.dot(w)
    risk = optimizer._compute_risk_metrics(port_returns, rf)

    logger.info("get_portfolio_metrics: return=%.2f%%, vol=%.2f%%", port_return * 100, port_vol * 100)

    return {
        "tickers": tickers,
        "weights": {t: round(v, 4) for t, v in weights.items()},
        "expected_annual_return": round(port_return, 4),
        "annual_volatility": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 2),
        "var_95_daily": round(risk["var_95"], 4),
        "cvar_95_daily": round(risk["cvar_95"], 4),
        "max_drawdown": round(risk["max_drawdown"], 4),
        "sortino_ratio": round(risk["sortino_ratio"], 2),
        "calmar_ratio": round(risk["calmar_ratio"], 2),
        "skewness": round(risk["skewness"], 3),
        "kurtosis": round(risk["kurtosis"], 3),
    }


def compare_strategies(context: dict, **kwargs) -> dict:
    """Compare Max Sharpe, Equal Weight, and Risk Parity strategies."""
    prices = context["prices"]
    optimizer = context["optimizer"]
    rf = context["risk_free_rate"]

    eq = optimizer.equal_weight(prices, rf)
    rp = optimizer.risk_parity(prices, rf)

    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252
    tickers = list(prices.columns)

    ms = optimizer._max_sharpe(mean_ret, cov, rf, tickers)

    logger.info("compare_strategies: 3 strategies computed")

    strategies = {}
    for name, p in [("Max Sharpe", ms), ("Equal Weight", eq), ("Risk Parity", rp)]:
        strategies[name] = {
            "expected_return": round(p.expected_return, 4),
            "volatility": round(p.volatility, 4),
            "sharpe_ratio": round(p.sharpe_ratio, 2),
            "weights": {t: round(v, 4) for t, v in p.weights.items()},
        }

    return {"strategies": strategies}


def run_backtest(context: dict, strategy: str = "current", **kwargs) -> dict:
    """Backtest a portfolio strategy over the historical period."""
    prices = context["prices"]
    optimizer = context["optimizer"]
    rf = context["risk_free_rate"]

    if strategy == "equal_weight":
        p = optimizer.equal_weight(prices, rf)
        weights = p.weights
    elif strategy == "risk_parity":
        p = optimizer.risk_parity(prices, rf)
        weights = p.weights
    elif strategy == "max_sharpe":
        returns = prices.pct_change().dropna()
        p = optimizer._max_sharpe(returns.mean() * 252, returns.cov() * 252,
                                  rf, list(prices.columns))
        weights = p.weights
    else:
        weights = context["weights"]

    bt = optimizer.backtest(prices, weights)

    logger.info("run_backtest: strategy=%s, total_return=%.2f%%", strategy, bt["total_return"] * 100)

    return {
        "strategy": strategy,
        "total_return": round(bt["total_return"], 4),
        "annualized_return": round(bt["annualized_return"], 4),
        "annualized_volatility": round(bt["annualized_vol"], 4),
        "max_drawdown": round(bt["max_drawdown"], 4),
        "sharpe_ratio": round(bt["sharpe"], 2),
    }


def get_correlation_analysis(context: dict, **kwargs) -> dict:
    """Analyze asset correlations and diversification."""
    prices = context["prices"]
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    # Find most and least correlated pairs
    tickers = list(corr.columns)
    pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append((tickers[i], tickers[j], round(float(corr.iloc[i, j]), 3)))

    pairs.sort(key=lambda x: x[2])
    least = pairs[0] if pairs else None
    most = pairs[-1] if pairs else None

    avg_corr = float(corr.values[np.triu_indices_from(corr.values, k=1)].mean())

    logger.info("get_correlation_analysis: avg_corr=%.3f", avg_corr)

    return {
        "correlation_matrix": {t: {t2: round(float(corr.loc[t, t2]), 3)
                                   for t2 in tickers} for t in tickers},
        "average_correlation": round(avg_corr, 3),
        "most_correlated_pair": {"pair": [most[0], most[1]], "correlation": most[2]} if most else None,
        "least_correlated_pair": {"pair": [least[0], least[1]], "correlation": least[2]} if least else None,
    }


# ── Tool Schemas (OpenAI function calling format) ────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_metrics",
            "description": "Get the current portfolio's risk and return metrics including expected return, volatility, Sharpe ratio, VaR, CVaR, max drawdown, Sortino ratio, and Calmar ratio.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_strategies",
            "description": "Compare three portfolio strategies side by side: Max Sharpe (Markowitz optimal), Equal Weight (1/N), and Risk Parity. Returns expected return, volatility, Sharpe ratio, and weights for each.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "Backtest a portfolio strategy over the historical data period. Returns total return, annualized return, annualized volatility, max drawdown, and Sharpe ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["current", "max_sharpe", "equal_weight", "risk_parity"],
                        "description": "Which strategy to backtest. 'current' uses the user's current weights.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation_analysis",
            "description": "Analyze pairwise correlations between portfolio assets. Returns the correlation matrix, average correlation, and the most/least correlated pairs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Registry mapping function names to callables
TOOL_REGISTRY = {
    "get_portfolio_metrics": get_portfolio_metrics,
    "compare_strategies": compare_strategies,
    "run_backtest": run_backtest,
    "get_correlation_analysis": get_correlation_analysis,
}
