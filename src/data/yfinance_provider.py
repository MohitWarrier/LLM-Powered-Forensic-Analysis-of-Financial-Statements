import logging

import pandas as pd
import yfinance as yf

from src.core.interfaces import DataProvider

logger = logging.getLogger(__name__)

# Popular preset portfolios for quick selection
PRESET_PORTFOLIOS = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Diversified ETFs": ["SPY", "QQQ", "IWM", "EFA", "AGG"],
    "FAANG + Tesla": ["META", "AAPL", "AMZN", "NFLX", "GOOGL", "TSLA"],
    "Sector ETFs": ["XLK", "XLF", "XLV", "XLE", "XLI"],
    "Global Mix": ["SPY", "EFA", "EEM", "GLD", "TLT"],
}


class YFinanceProvider(DataProvider):
    """Fetches real market data from Yahoo Finance."""

    def __init__(self, tickers: list[str] = None, period: str = "2y"):
        self._tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self._period = period
        self._cache: dict[str, pd.DataFrame] = {}

    def set_tickers(self, tickers: list[str]):
        self._tickers = tickers
        self._cache.clear()

    def set_period(self, period: str):
        self._period = period
        self._cache.clear()

    def get_forensic_data(self) -> pd.DataFrame:
        raise NotImplementedError("YFinance forensic data not yet implemented")

    def get_portfolio_data(self) -> pd.DataFrame:
        cache_key = f"{','.join(sorted(self._tickers))}_{self._period}"
        if cache_key in self._cache:
            logger.info("Returning cached price data for %s", self._tickers)
            return self._cache[cache_key]

        logger.info("Fetching price data from Yahoo Finance: tickers=%s, period=%s",
                     self._tickers, self._period)
        try:
            data = yf.download(
                self._tickers,
                period=self._period,
                auto_adjust=True,
                progress=False,
            )

            # yf.download returns multi-level columns when multiple tickers
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"]
            else:
                # Single ticker case
                prices = data[["Close"]]
                prices.columns = self._tickers

            # Drop any tickers that returned all NaN
            prices = prices.dropna(axis=1, how="all")
            # Forward-fill then drop remaining NaN rows
            prices = prices.ffill().dropna()

            if prices.empty:
                raise ValueError("No price data returned. Check ticker symbols.")

            failed = set(self._tickers) - set(prices.columns)
            if failed:
                logger.warning("Failed to fetch data for: %s", failed)

            logger.info("Fetched price data: %d rows, %d tickers, date range %s to %s",
                         len(prices), len(prices.columns),
                         prices.index[0].strftime("%Y-%m-%d"),
                         prices.index[-1].strftime("%Y-%m-%d"))

            self._cache[cache_key] = prices
            return prices

        except Exception as e:
            logger.error("Failed to fetch data from Yahoo Finance: %s", e)
            raise


def validate_tickers(tickers: list[str]) -> dict[str, bool]:
    """Quick check which tickers are valid by attempting a tiny download."""
    results = {}
    for t in tickers:
        try:
            info = yf.Ticker(t)
            hist = info.history(period="5d")
            results[t] = len(hist) > 0
        except Exception:
            results[t] = False
    return results