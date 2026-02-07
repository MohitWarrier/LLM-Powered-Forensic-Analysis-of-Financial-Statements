import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application-wide configuration."""

    APP_NAME = "Financial Analysis Suite"
    APP_VERSION = "0.2.0"

    # Feature flags
    FEATURES = {
        "forensic_analysis": True,
        "portfolio_optimizer": True,
        "quant_analyst": True,
    }

    # Data provider: "dummy", "csv", "api"
    DATA_PROVIDER = os.getenv("DATA_PROVIDER", "dummy")

    # LLM configuration
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Portfolio optimizer defaults
    PORTFOLIO_RISK_FREE_RATE = 0.02
    PORTFOLIO_NUM_SIMULATIONS = 5000


settings = Settings()
