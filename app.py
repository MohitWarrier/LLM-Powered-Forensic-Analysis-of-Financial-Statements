import logging
import streamlit as st

#prompt $N$G
#prompt $P$G

# Configure logging for all modules — must be before any src imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from config import settings
from src.core.registry import FeatureRegistry, Feature
from src.data.dummy_provider import DummyDataProvider
from src.features.forensic.service import ForensicService
from src.features.forensic.page import render_forensic_page
from src.features.portfolio.service import PortfolioService
from src.features.portfolio.page import render_portfolio_page
from src.features.quant_analyst.service import QuantAnalystService
from src.features.quant_analyst.page import render_quant_analyst_page
from src.ui.layout import render_sidebar, render_home

st.set_page_config(
    page_title="PortfolioLab",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


@st.cache_resource
def build_registry() -> FeatureRegistry:
    """Build the feature registry once, cached across Streamlit reruns."""
    logger.info("Building feature registry...")
    data_provider = DummyDataProvider()
    registry = FeatureRegistry()

    if settings.FEATURES.get("forensic_analysis"):
        svc = ForensicService(data_provider)
        registry.register(Feature(
            id="forensic",
            name="Forensic Analysis",
            icon="mag",
            service=svc,
            page_renderer=render_forensic_page,
        ))
        logger.info("Registered feature: Forensic Analysis")

    if settings.FEATURES.get("portfolio_optimizer"):
        svc = PortfolioService(data_provider)
        registry.register(Feature(
            id="portfolio",
            name="Portfolio Optimizer",
            icon="chart_with_upwards_trend",
            service=svc,
            page_renderer=render_portfolio_page,
        ))
        logger.info("Registered feature: Portfolio Optimizer")

    if settings.FEATURES.get("quant_analyst"):
        svc = QuantAnalystService(data_provider)
        registry.register(Feature(
            id="quant_analyst",
            name="AI Quant Analyst",
            icon="robot_face",
            service=svc,
            page_renderer=render_quant_analyst_page,
        ))
        logger.info("Registered feature: AI Quant Analyst")

    logger.info("Registry ready: %d features enabled", len(registry.get_enabled()))
    return registry


registry = build_registry()

# Sidebar navigation
selected = render_sidebar(registry)
logger.info("Page selected: %s", selected)

# Route to the selected page
if selected == "home":
    render_home(registry)
else:
    feature = registry.get(selected)
    feature.page_renderer(feature.service)
