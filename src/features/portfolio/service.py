import logging

import pandas as pd

from src.core.interfaces import BaseService, DataProvider
from src.features.portfolio.optimizer import MeanVarianceOptimizer

logger = logging.getLogger(__name__)


class PortfolioService(BaseService):
    """Orchestrates portfolio optimization."""

    def __init__(self, data_provider: DataProvider):
        self._data_provider = data_provider
        self._yfinance_provider = None
        self._optimizer = MeanVarianceOptimizer()
        logger.info("PortfolioService initialized")

    def get_name(self) -> str:
        return "Portfolio Optimizer"

    def get_description(self) -> str:
        return "Mean-variance portfolio optimization with efficient frontier"

    def get_sample_data(self) -> pd.DataFrame:
        logger.info("Loading sample portfolio data")
        return self._data_provider.get_portfolio_data()

    def get_live_data(self, tickers: list[str], period: str = "2y") -> pd.DataFrame:
        """Fetch live market data via Yahoo Finance."""
        if self._yfinance_provider is None:
            from src.data.yfinance_provider import YFinanceProvider
            self._yfinance_provider = YFinanceProvider(tickers, period)
        else:
            self._yfinance_provider.set_tickers(tickers)
            self._yfinance_provider.set_period(period)
        logger.info("Fetching live data: tickers=%s, period=%s", tickers, period)
        return self._yfinance_provider.get_portfolio_data()

    def optimize(self, prices: pd.DataFrame, risk_free_rate: float = 0.02,
                 num_portfolios: int = 5000,
                 weight_bounds: tuple = (0.0, 1.0)) -> dict:
        logger.info("Starting optimization: %d assets, rf=%.2f%%, sims=%d, bounds=%s",
                     len(prices.columns), risk_free_rate * 100, num_portfolios, weight_bounds)
        result = self._optimizer.optimize(prices, risk_free_rate=risk_free_rate,
                                          num_portfolios=num_portfolios,
                                          weight_bounds=weight_bounds)
        logger.info("Optimization complete: optimal sharpe=%.2f, min_var vol=%.4f",
                     result["optimal"].sharpe_ratio, result["min_variance"].volatility)
        return result

    def compare_strategies(self, prices: pd.DataFrame,
                           risk_free_rate: float = 0.02) -> dict:
        """Run all three strategies and return comparison dict."""
        logger.info("Comparing strategies on %d assets", len(prices.columns))
        optimal = self._optimizer._max_sharpe(
            prices.pct_change().dropna().mean() * 252,
            prices.pct_change().dropna().cov() * 252,
            risk_free_rate, list(prices.columns),
        )
        equal_wt = self._optimizer.equal_weight(prices, risk_free_rate)
        risk_par = self._optimizer.risk_parity(prices, risk_free_rate)
        logger.info("Strategy comparison complete")
        return {
            "Max Sharpe": optimal,
            "Equal Weight": equal_wt,
            "Risk Parity": risk_par,
        }

    def backtest(self, prices: pd.DataFrame, weights: dict[str, float],
                 benchmark_ticker: str = None) -> dict:
        """Backtest a portfolio allocation."""
        logger.info("Running backtest with %d assets", len(weights))
        return self._optimizer.backtest(prices, weights, benchmark_ticker)

    def get_available_tickers(self) -> list[str]:
        data = self._data_provider.get_portfolio_data()
        tickers = list(data.columns)
        logger.info("Available tickers: %s", tickers)
        return tickers