import logging

import pandas as pd

from src.core.interfaces import BaseService, DataProvider
from src.features.forensic.analyzer import ForensicAnalyzer
from src.llm.formatter import prompt_builder
from src.llm.groq_client import chat_with_groq

logger = logging.getLogger(__name__)

REQUIRED_METRICS = [
    "Net Income", "Operating Cash Flow", "Interest Income",
    "Cash and Cash Equivalents", "Contingent Liabilities",
    "Shareholder Equity", "Total Debt"
]


class ForensicService(BaseService):
    """Orchestrates forensic analysis: data loading, analysis, LLM explanation."""

    def __init__(self, data_provider: DataProvider):
        self._data_provider = data_provider
        self._analyzer = ForensicAnalyzer()
        logger.info("ForensicService initialized")

    def get_name(self) -> str:
        return "Forensic Analysis"

    def get_description(self) -> str:
        return "Detect financial red flags in company statements"

    def get_sample_data(self) -> pd.DataFrame:
        logger.info("Loading sample forensic data")
        return self._data_provider.get_forensic_data()

    def analyze(self, df: pd.DataFrame) -> dict:
        logger.info("Starting forensic analysis on %d metrics, %d years", len(df.index), len(df.columns))
        return self._analyzer.analyze(df)

    def explain(self, results: dict) -> str:
        logger.info("Generating LLM explanation for analysis results")
        prompt = prompt_builder(results)
        return chat_with_groq(prompt)

    def validate_data(self, df: pd.DataFrame) -> list[str]:
        """Return list of missing required metrics."""
        missing = [m for m in REQUIRED_METRICS if m not in df.index]
        if missing:
            logger.warning("Data validation failed, missing: %s", missing)
        return missing
