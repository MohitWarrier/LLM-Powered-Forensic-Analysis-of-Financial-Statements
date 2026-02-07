from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    """Abstract data source. Implement to add a new data source."""

    @abstractmethod
    def get_forensic_data(self) -> pd.DataFrame:
        """Return financial statement data.
        DataFrame: metrics as index, years as columns.
        """
        ...

    @abstractmethod
    def get_portfolio_data(self) -> pd.DataFrame:
        """Return historical price data for portfolio optimization.
        DataFrame: dates as index, tickers as columns, close prices as values.
        """
        ...


class AnalyzerInterface(ABC):
    """Contract for any analysis module."""

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        """Run analysis on financial data, return structured results."""
        ...


class OptimizerInterface(ABC):
    """Contract for any optimization module."""

    @abstractmethod
    def optimize(self, data: pd.DataFrame, **params) -> dict:
        """Run optimization, return structured results."""
        ...


class BaseService(ABC):
    """Base for feature services."""

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def get_description(self) -> str:
        ...
