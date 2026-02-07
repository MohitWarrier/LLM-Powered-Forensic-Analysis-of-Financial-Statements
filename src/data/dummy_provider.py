import logging

import numpy as np
import pandas as pd

from src.core.interfaces import DataProvider

logger = logging.getLogger(__name__)


class DummyDataProvider(DataProvider):
    """Hardcoded sample data for development and demo."""

    def get_forensic_data(self) -> pd.DataFrame:
        logger.info("Generating dummy forensic data (2 years, 8 metrics)")
        data = {
            "2022": {
                "Revenue": 2000000,
                "Net Income": 800000,
                "Operating Cash Flow": 350000,
                "Interest Income": 4000,
                "Cash and Cash Equivalents": 800000,
                "Contingent Liabilities": 120000,
                "Shareholder Equity": 900000,
                "Total Debt": 1300000,
            },
            "2023": {
                "Revenue": 2600000,
                "Net Income": 1000000,
                "Operating Cash Flow": 400000,
                "Interest Income": 5000,
                "Cash and Cash Equivalents": 1000000,
                "Contingent Liabilities": 150000,
                "Shareholder Equity": 1000000,
                "Total Debt": 1500000,
            },
        }
        return pd.DataFrame(data)

    def get_portfolio_data(self) -> pd.DataFrame:
        """Generate synthetic daily price data for 5 assets over 2 years."""
        np.random.seed(42)
        dates = pd.bdate_range(start="2022-01-03", end="2023-12-29")
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        logger.info("Generating dummy portfolio data: %d days, tickers=%s", len(dates), tickers)

        annual_returns = [0.12, 0.10, 0.14, 0.08, 0.20]
        annual_vols = [0.25, 0.22, 0.20, 0.28, 0.45]

        prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        for i, ticker in enumerate(tickers):
            daily_ret = annual_returns[i] / 252
            daily_vol = annual_vols[i] / np.sqrt(252)
            returns = np.random.normal(daily_ret, daily_vol, len(dates))
            prices[ticker] = 100 * np.cumprod(1 + returns)

        return prices
