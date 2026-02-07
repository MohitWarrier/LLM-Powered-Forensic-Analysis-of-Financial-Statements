from dataclasses import dataclass


@dataclass
class PortfolioResult:
    """Result of a single portfolio calculation."""
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
