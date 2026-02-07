import pandas as pd

from src.core.interfaces import DataProvider


class CSVDataProvider(DataProvider):
    """Wraps user-uploaded CSV/Excel DataFrames."""

    def __init__(self, forensic_df: pd.DataFrame = None,
                 portfolio_df: pd.DataFrame = None):
        self._forensic_df = forensic_df
        self._portfolio_df = portfolio_df

    def set_forensic_data(self, df: pd.DataFrame):
        self._forensic_df = df

    def set_portfolio_data(self, df: pd.DataFrame):
        self._portfolio_df = df

    def get_forensic_data(self) -> pd.DataFrame:
        if self._forensic_df is None:
            raise ValueError("No forensic data loaded. Upload a file first.")
        return self._forensic_df

    def get_portfolio_data(self) -> pd.DataFrame:
        if self._portfolio_df is None:
            raise ValueError("No portfolio data loaded. Upload a file first.")
        return self._portfolio_df
