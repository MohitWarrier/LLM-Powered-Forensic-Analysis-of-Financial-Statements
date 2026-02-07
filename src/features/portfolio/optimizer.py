import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.core.types import PortfolioResult

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """
    Markowitz mean-variance portfolio optimization.

    Computes:
    - Expected returns from historical data (annualized mean daily returns)
    - Covariance matrix from historical daily returns
    - Efficient frontier via Monte Carlo simulation
    - Optimal portfolio (max Sharpe ratio) via scipy.optimize
    - Minimum-variance portfolio via scipy.optimize
    """

    def optimize(self, prices: pd.DataFrame, risk_free_rate: float = 0.02,
                 num_portfolios: int = 5000, weight_bounds: tuple = (0.0, 1.0)) -> dict:
        returns = prices.pct_change().dropna()
        tickers = list(prices.columns)
        n_assets = len(tickers)

        logger.info("Computing returns: %d days, %d assets", len(returns), n_assets)

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        logger.info("Annualized returns: %s", {t: f"{r:.2%}" for t, r in mean_returns.items()})

        # Monte Carlo simulation for efficient frontier
        logger.info("Running Monte Carlo simulation with %d portfolios...", num_portfolios)
        frontier = []
        for _ in range(num_portfolios):
            w = np.random.dirichlet(np.ones(n_assets))
            w = np.clip(w, weight_bounds[0], weight_bounds[1])
            w = w / w.sum()  # re-normalize
            ret = float(np.dot(w, mean_returns))
            vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            frontier.append(PortfolioResult(
                weights=dict(zip(tickers, w.round(4).tolist())),
                expected_return=ret,
                volatility=vol,
                sharpe_ratio=float(sharpe),
            ))

        # Max Sharpe ratio portfolio
        logger.info("Optimizing for max Sharpe ratio...")
        optimal = self._max_sharpe(mean_returns, cov_matrix, risk_free_rate, tickers, weight_bounds)
        logger.info("Max Sharpe portfolio: return=%.2f%%, vol=%.2f%%, sharpe=%.2f",
                     optimal.expected_return * 100, optimal.volatility * 100, optimal.sharpe_ratio)

        # Minimum variance portfolio
        logger.info("Optimizing for minimum variance...")
        min_var = self._min_variance(mean_returns, cov_matrix, risk_free_rate, tickers, weight_bounds)
        logger.info("Min variance portfolio: return=%.2f%%, vol=%.2f%%, sharpe=%.2f",
                     min_var.expected_return * 100, min_var.volatility * 100, min_var.sharpe_ratio)

        # Individual asset stats
        individual = {}
        for t in tickers:
            individual[t] = {
                "expected_return": float(mean_returns[t]),
                "volatility": float(np.sqrt(cov_matrix.loc[t, t])),
            }

        # Correlation matrix
        corr_matrix = returns.corr()

        # Risk metrics for the optimal portfolio
        optimal_weights = np.array(list(optimal.weights.values()))
        portfolio_returns = returns.dot(optimal_weights)
        risk_metrics = self._compute_risk_metrics(portfolio_returns, risk_free_rate)

        # Cumulative returns for each asset (for price chart)
        cumulative_returns = (1 + returns).cumprod()

        logger.info("Risk metrics: VaR_95=%.2f%%, CVaR_95=%.2f%%, max_drawdown=%.2f%%",
                     risk_metrics["var_95"] * 100, risk_metrics["cvar_95"] * 100,
                     risk_metrics["max_drawdown"] * 100)

        return {
            "optimal": optimal,
            "min_variance": min_var,
            "efficient_frontier": frontier,
            "individual_stats": individual,
            "correlation_matrix": corr_matrix,
            "risk_metrics": risk_metrics,
            "cumulative_returns": cumulative_returns,
        }

    def _compute_risk_metrics(self, portfolio_returns: pd.Series, risk_free_rate: float) -> dict:
        """Compute VaR, CVaR, max drawdown, Sortino ratio for the optimal portfolio."""
        # Value at Risk (95%)
        var_95 = float(np.percentile(portfolio_returns, 5))

        # Conditional VaR (Expected Shortfall)
        cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Sortino ratio (downside deviation)
        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0
        ann_return = float(portfolio_returns.mean() * 252)
        sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "skewness": float(portfolio_returns.skew()),
            "kurtosis": float(portfolio_returns.kurtosis()),
        }

    def _max_sharpe(self, mean_returns, cov_matrix, rf, tickers,
                    weight_bounds=(0.0, 1.0)) -> PortfolioResult:
        n = len(tickers)

        def neg_sharpe(w):
            ret = np.dot(w, mean_returns.values)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
            return -(ret - rf) / vol if vol > 0 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple(weight_bounds for _ in range(n))
        x0 = np.array([1.0 / n] * n)

        result = minimize(neg_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        w = result.x
        ret = float(np.dot(w, mean_returns.values))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
        sharpe = (ret - rf) / vol if vol > 0 else 0

        return PortfolioResult(
            weights=dict(zip(tickers, w.round(4).tolist())),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=float(sharpe),
        )

    def _min_variance(self, mean_returns, cov_matrix, rf, tickers,
                      weight_bounds=(0.0, 1.0)) -> PortfolioResult:
        n = len(tickers)

        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple(weight_bounds for _ in range(n))
        x0 = np.array([1.0 / n] * n)

        result = minimize(portfolio_vol, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        w = result.x
        ret = float(np.dot(w, mean_returns.values))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
        sharpe = (ret - rf) / vol if vol > 0 else 0

        return PortfolioResult(
            weights=dict(zip(tickers, w.round(4).tolist())),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=float(sharpe),
        )

    def equal_weight(self, prices: pd.DataFrame, risk_free_rate: float = 0.02) -> PortfolioResult:
        """Simple 1/N allocation."""
        returns = prices.pct_change().dropna()
        tickers = list(prices.columns)
        n = len(tickers)
        w = np.array([1.0 / n] * n)

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        ret = float(np.dot(w, mean_returns.values))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

        logger.info("Equal weight portfolio: return=%.2f%%, vol=%.2f%%, sharpe=%.2f",
                    ret * 100, vol * 100, sharpe)

        return PortfolioResult(
            weights=dict(zip(tickers, w.round(4).tolist())),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=float(sharpe),
        )

    def risk_parity(self, prices: pd.DataFrame, risk_free_rate: float = 0.02) -> PortfolioResult:
        """Allocate so each asset contributes equally to portfolio risk."""
        returns = prices.pct_change().dropna()
        tickers = list(prices.columns)
        n = len(tickers)
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252

        # Inverse volatility as starting point
        vols = np.sqrt(np.diag(cov_matrix.values))
        inv_vols = 1.0 / vols
        w = inv_vols / inv_vols.sum()

        # Iterative risk parity optimization
        def risk_budget_objective(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
            marginal_contrib = np.dot(cov_matrix.values, w)
            risk_contrib = w * marginal_contrib / port_vol
            target_risk = port_vol / n
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 1.0) for _ in range(n))

        result = minimize(risk_budget_objective, w, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        w = result.x
        w = w / w.sum()  # ensure sums to 1

        ret = float(np.dot(w, mean_returns.values))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

        logger.info("Risk parity portfolio: return=%.2f%%, vol=%.2f%%, sharpe=%.2f",
                    ret * 100, vol * 100, sharpe)

        return PortfolioResult(
            weights=dict(zip(tickers, w.round(4).tolist())),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=float(sharpe),
        )

    def backtest(self, prices: pd.DataFrame, weights: dict[str, float],
                 benchmark_ticker: str = None) -> dict:
        """Backtest a portfolio allocation against a benchmark.

        Returns dict with:
            portfolio_value: pd.Series of portfolio value over time (starts at 1.0)
            benchmark_value: pd.Series of benchmark value (if provided)
            total_return: float
            annualized_return: float
            annualized_vol: float
            max_drawdown: float
            sharpe: float
        """
        returns = prices.pct_change().dropna()
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])

        portfolio_returns = returns[tickers].dot(w)
        portfolio_value = (1 + portfolio_returns).cumprod()

        # Metrics
        total_return = float(portfolio_value.iloc[-1] - 1)
        n_days = len(portfolio_returns)
        ann_return = float((1 + total_return) ** (252 / n_days) - 1)
        ann_vol = float(portfolio_returns.std() * np.sqrt(252))

        # Max drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Sharpe (assume 2% rf)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        result = {
            "portfolio_value": portfolio_value,
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        }

        # Benchmark (equal weight of all assets as default)
        if benchmark_ticker and benchmark_ticker in prices.columns:
            bench_returns = returns[benchmark_ticker]
            result["benchmark_value"] = (1 + bench_returns).cumprod()

        logger.info("Backtest complete: total_return=%.2f%%, ann_return=%.2f%%, max_dd=%.2f%%",
                    total_return * 100, ann_return * 100, max_dd * 100)

        return result
