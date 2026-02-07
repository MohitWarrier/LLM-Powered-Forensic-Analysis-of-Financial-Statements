import logging

import streamlit as st
import pandas as pd

from src.features.quant_analyst.service import QuantAnalystService
from src.data.yfinance_provider import PRESET_PORTFOLIOS
from src.llm.groq_client import is_llm_available

logger = logging.getLogger(__name__)

SUGGESTED_QUESTIONS = [
    "What's my portfolio's risk profile?",
    "Compare my options — Max Sharpe vs Equal Weight vs Risk Parity",
    "How correlated are my assets?",
    "Backtest my current allocation",
]


def render_quant_analyst_page(service: QuantAnalystService):
    """Streamlit page for the AI Quant Analyst."""
    st.title("AI Quant Analyst")
    st.markdown("Ask questions about your portfolio in natural language")

    if not is_llm_available():
        st.warning("This feature requires a Groq API key. Set `OPENAI_API_KEY` in your `.env` file.")
        st.stop()

    # ── Portfolio Setup ──────────────────────────────────────────
    prices = _render_portfolio_setup(service)
    if prices is None:
        st.stop()

    # Compute default weights (equal weight) if not set
    if "qa_weights" not in st.session_state:
        tickers = list(prices.columns)
        st.session_state["qa_weights"] = {t: round(1.0 / len(tickers), 4) for t in tickers}

    weights = st.session_state["qa_weights"]

    st.caption(f"Portfolio: {', '.join(weights.keys())} | Weights: {', '.join(f'{v:.0%}' for v in weights.values())}")
    st.markdown("---")

    # ── Chat Interface ───────────────────────────────────────────
    if "qa_chat_history" not in st.session_state:
        st.session_state["qa_chat_history"] = []
    if "qa_display_messages" not in st.session_state:
        st.session_state["qa_display_messages"] = []

    # Display chat history
    for msg in st.session_state["qa_display_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                with st.expander("Tools used"):
                    for t in msg["tools_used"]:
                        st.caption(f"Called `{t['tool']}`")

    # Suggested questions (only show if no chat history)
    if not st.session_state["qa_display_messages"]:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"suggest_{i}"):
                    _handle_user_message(service, q, prices, weights)
                    st.rerun()

    # Chat input
    user_input = st.chat_input("Ask about your portfolio...")
    if user_input:
        _handle_user_message(service, user_input, prices, weights)
        st.rerun()


def _handle_user_message(service: QuantAnalystService, message: str,
                         prices: pd.DataFrame, weights: dict):
    """Process a user message through the LLM tool-use loop."""
    # Add user message to display
    st.session_state["qa_display_messages"].append({
        "role": "user", "content": message,
    })

    # Build chat history for the LLM (just role + content, no display metadata)
    chat_history = []
    for msg in st.session_state["qa_chat_history"]:
        chat_history.append({"role": msg["role"], "content": msg["content"]})

    try:
        response, tools_used = service.run_chat(
            user_message=message,
            chat_history=chat_history,
            prices=prices,
            weights=weights,
        )

        # Store in chat history for future context
        st.session_state["qa_chat_history"].append({"role": "user", "content": message})
        st.session_state["qa_chat_history"].append({"role": "assistant", "content": response})

        # Store in display messages
        st.session_state["qa_display_messages"].append({
            "role": "assistant",
            "content": response,
            "tools_used": tools_used,
        })

        logger.info("Chat response generated, %d tools used", len(tools_used))

    except Exception as e:
        logger.error("Chat failed: %s", e)
        st.session_state["qa_display_messages"].append({
            "role": "assistant",
            "content": f"Sorry, I encountered an error: {e}",
        })


def _render_portfolio_setup(service: QuantAnalystService) -> pd.DataFrame | None:
    """Render portfolio data loading UI. Returns prices DataFrame or None."""
    use_live = st.toggle("Use live market data", value=False, key="qa_live_toggle")

    if not use_live:
        prices = service.get_sample_data()
        st.caption(f"Sample data: {len(prices)} days, {list(prices.columns)}")
        st.session_state["qa_prices"] = prices
        return prices

    # Live data UI
    col1, col2 = st.columns([3, 1])
    with col1:
        preset = st.selectbox("Preset:", ["Custom"] + list(PRESET_PORTFOLIOS.keys()),
                              key="qa_preset")
    with col2:
        period = st.selectbox("Period:", ["6mo", "1y", "2y", "5y"], index=2, key="qa_period")

    default_val = ", ".join(PRESET_PORTFOLIOS[preset]) if preset != "Custom" else "AAPL, MSFT, GOOGL, AMZN, META"
    ticker_input = st.text_input("Tickers:", value=default_val, key="qa_tickers")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers.")
        return None

    cache_key = f"qa_{','.join(sorted(tickers))}_{period}"
    cached = st.session_state.get("qa_prices")
    if st.session_state.get("qa_cache_key") == cache_key and cached is not None:
        st.caption(f"{len(cached)} days | {cached.index[0].strftime('%Y-%m-%d')} to {cached.index[-1].strftime('%Y-%m-%d')}")
        return cached

    if st.button("Fetch Data", key="qa_fetch"):
        with st.spinner(f"Fetching {', '.join(tickers)}..."):
            try:
                prices = service.get_live_data(tickers, period)
                st.session_state["qa_prices"] = prices
                st.session_state["qa_cache_key"] = cache_key
                # Reset weights for new data
                st.session_state["qa_weights"] = {t: round(1.0 / len(prices.columns), 4)
                                                   for t in prices.columns}
                st.session_state["qa_chat_history"] = []
                st.session_state["qa_display_messages"] = []
                st.rerun()
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                return None

    st.info("Click **Fetch Data** to load market prices.")
    return st.session_state.get("qa_prices")
