import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.features.portfolio.service import PortfolioService
from src.data.yfinance_provider import PRESET_PORTFOLIOS

logger = logging.getLogger(__name__)


def render_portfolio_page(service: PortfolioService):
    """Full Streamlit page for portfolio optimization."""
    st.title("Portfolio Optimizer")
    st.markdown("Mean-variance (Markowitz) portfolio optimization with efficient frontier")

    # ── Data Source ──────────────────────────────────────────────
    data_source = st.toggle("Use live market data (Yahoo Finance)", value=False)

    if data_source:
        prices = _load_live_data(service)
        if prices is None:
            st.stop()
    else:
        prices = service.get_sample_data()

    # ── Parameters ───────────────────────────────────────────────
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 10.0, 2.0, step=0.1) / 100
    with col2:
        num_portfolios = st.slider("Simulation count", 1000, 20000, 5000, step=1000)

    # Ticker selection from loaded data
    tickers = list(prices.columns)
    selected_tickers = st.multiselect(
        "Select assets to include:",
        tickers,
        default=tickers,
    )

    if len(selected_tickers) < 2:
        st.warning("Select at least 2 assets for portfolio optimization.")
        st.stop()

    # Weight constraints (collapsed)
    with st.expander("Weight Constraints"):
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.slider("Min weight per asset (%)", 0, 25, 0, step=1) / 100
        with col2:
            max_weight = st.slider("Max weight per asset (%)", 10, 100, 100, step=5) / 100
        if min_weight >= max_weight:
            st.error("Min weight must be less than max weight.")
            st.stop()
    weight_bounds = (min_weight, max_weight)

    # ── Session state ────────────────────────────────────────────
    if "portfolio_result" not in st.session_state:
        st.session_state["portfolio_result"] = None

    # ── Optimize ─────────────────────────────────────────────────
    if st.button("Optimize Portfolio", type="primary"):
        with st.spinner("Running optimization..."):
            logger.info("User triggered optimization: tickers=%s, rf=%.2f%%, sims=%d",
                         selected_tickers, risk_free_rate * 100, num_portfolios)
            p = prices[selected_tickers]
            result = service.optimize(p, risk_free_rate, num_portfolios, weight_bounds)
            st.session_state["portfolio_result"] = result
            st.session_state["portfolio_prices"] = p
            st.session_state["portfolio_rf"] = risk_free_rate
            st.session_state.pop("portfolio_strategies", None)
            st.session_state.pop("portfolio_backtests", None)
            logger.info("Optimization complete")

    result = st.session_state.get("portfolio_result")
    if not result:
        return

    # ── Results ──────────────────────────────────────────────────
    _render_optimal_portfolio(result)
    _render_risk_metrics(result)
    _render_efficient_frontier(result)
    _render_weights_chart(result)
    _render_correlation_heatmap(result)
    _render_cumulative_returns(result)
    _render_individual_stats(result)

    # ── Strategy Comparison (opt-in) ─────────────────────────────
    st.markdown("---")
    with st.expander("Compare Strategies (Max Sharpe vs Equal Weight vs Risk Parity)"):
        if st.button("Run Strategy Comparison"):
            with st.spinner("Comparing strategies..."):
                p = st.session_state.get("portfolio_prices", prices[selected_tickers])
                rf = st.session_state.get("portfolio_rf", risk_free_rate)
                strategies = service.compare_strategies(p, rf)
                st.session_state["portfolio_strategies"] = strategies
        strategies = st.session_state.get("portfolio_strategies")
        if strategies:
            _render_strategy_comparison(strategies)

    # ── Backtest (opt-in) ────────────────────────────────────────
    with st.expander("Backtest Portfolio"):
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                p = st.session_state.get("portfolio_prices", prices[selected_tickers])
                strategies = st.session_state.get("portfolio_strategies")
                if strategies:
                    backtests = {}
                    for name, portfolio in strategies.items():
                        backtests[name] = service.backtest(p, portfolio.weights)
                else:
                    backtests = {"Optimal (Max Sharpe)": service.backtest(p, result["optimal"].weights)}
                st.session_state["portfolio_backtests"] = backtests
        backtests = st.session_state.get("portfolio_backtests")
        if backtests:
            _render_backtest(backtests)

    # ── Export ────────────────────────────────────────────────────
    st.markdown("---")
    _render_export(result, st.session_state.get("portfolio_strategies"),
                   st.session_state.get("portfolio_backtests"))


def _load_live_data(service: PortfolioService) -> pd.DataFrame | None:
    """Inline UI for loading live Yahoo Finance data."""
    col1, col2 = st.columns([3, 1])
    with col1:
        preset = st.selectbox(
            "Preset portfolio:",
            ["Custom"] + list(PRESET_PORTFOLIOS.keys()),
        )
    with col2:
        period = st.selectbox("Period:", ["6mo", "1y", "2y", "5y"], index=2)

    default_val = ", ".join(PRESET_PORTFOLIOS[preset]) if preset != "Custom" else "AAPL, MSFT, GOOGL, AMZN, META"
    ticker_input = st.text_input("Tickers (comma-separated):", value=default_val)
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers.")
        return None

    cache_key = f"{','.join(sorted(tickers))}_{period}"
    cached = st.session_state.get("live_prices")
    if st.session_state.get("live_cache_key") == cache_key and cached is not None:
        st.caption(f"{len(cached)} days | {cached.index[0].strftime('%Y-%m-%d')} to {cached.index[-1].strftime('%Y-%m-%d')}")
        return cached

    if st.button("Fetch Market Data"):
        with st.spinner(f"Fetching {', '.join(tickers)}..."):
            try:
                prices = service.get_live_data(tickers, period)
                st.session_state["live_prices"] = prices
                st.session_state["live_cache_key"] = cache_key
                st.session_state.pop("portfolio_result", None)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                logger.error("Live data fetch failed: %s", e)
                return None

    st.info("Click **Fetch Market Data** to load prices, then optimize.")
    return None


# ═══════════════════════════════════════════════════════════════════
# Renderers
# ═══════════════════════════════════════════════════════════════════

def _render_optimal_portfolio(result: dict):
    st.subheader("Optimal Portfolio (Max Sharpe Ratio)")
    optimal = result["optimal"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Annual Return", f"{optimal.expected_return:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{optimal.volatility:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{optimal.sharpe_ratio:.2f}")

    st.subheader("Minimum Variance Portfolio")
    min_var = result["min_variance"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Annual Return", f"{min_var.expected_return:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{min_var.volatility:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{min_var.sharpe_ratio:.2f}")


def _render_risk_metrics(result: dict):
    st.subheader("Risk Analytics (Optimal Portfolio)")
    rm = result["risk_metrics"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VaR (95%, daily)", f"{rm['var_95']:.2%}")
    with col2:
        st.metric("CVaR (95%, daily)", f"{rm['cvar_95']:.2%}")
    with col3:
        st.metric("Max Drawdown", f"{rm['max_drawdown']:.2%}")
    with col4:
        st.metric("Sortino Ratio", f"{rm['sortino_ratio']:.2f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Calmar Ratio", f"{rm['calmar_ratio']:.2f}")
    with col2:
        st.metric("Skewness", f"{rm['skewness']:.3f}")
    with col3:
        st.metric("Kurtosis", f"{rm['kurtosis']:.3f}")


def _render_efficient_frontier(result: dict):
    st.subheader("Efficient Frontier")
    frontier = result["efficient_frontier"]
    vols = [p.volatility for p in frontier]
    rets = [p.expected_return for p in frontier]
    sharpes = [p.sharpe_ratio for p in frontier]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(color=sharpes, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Sharpe"), size=4, opacity=0.6),
        text=[f"Sharpe: {s:.2f}" for s in sharpes],
        name="Simulated Portfolios",
    ))

    opt = result["optimal"]
    fig.add_trace(go.Scatter(
        x=[opt.volatility], y=[opt.expected_return], mode="markers",
        marker=dict(color="red", size=16, symbol="star"),
        name=f"Max Sharpe ({opt.sharpe_ratio:.2f})",
    ))

    mv = result["min_variance"]
    fig.add_trace(go.Scatter(
        x=[mv.volatility], y=[mv.expected_return], mode="markers",
        marker=dict(color="blue", size=14, symbol="diamond"),
        name=f"Min Variance ({mv.sharpe_ratio:.2f})",
    ))

    for ticker, stats in result["individual_stats"].items():
        fig.add_trace(go.Scatter(
            x=[stats["volatility"]], y=[stats["expected_return"]],
            mode="markers+text",
            marker=dict(color="orange", size=10, symbol="circle"),
            text=[ticker], textposition="top center",
            name=ticker, showlegend=False,
        ))

    fig.update_layout(
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Expected Return",
        height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_weights_chart(result: dict):
    st.subheader("Optimal Portfolio Weights")
    weights = result["optimal"].weights
    filtered = {k: v for k, v in weights.items() if v > 0.001}
    fig = go.Figure(go.Bar(
        x=list(filtered.keys()), y=list(filtered.values()),
        text=[f"{v:.1%}" for v in filtered.values()],
        textposition="auto", marker_color="steelblue",
    ))
    fig.update_layout(yaxis_title="Weight", yaxis_tickformat=".0%", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_heatmap(result: dict):
    st.subheader("Asset Correlation Matrix")
    fig = px.imshow(
        result["correlation_matrix"], text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def _render_cumulative_returns(result: dict):
    st.subheader("Cumulative Returns")
    cum_ret = result["cumulative_returns"]
    fig = go.Figure()
    for col in cum_ret.columns:
        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret[col], mode="lines", name=col))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative Return (1 = start)",
        height=450, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_individual_stats(result: dict):
    st.subheader("Individual Asset Statistics")
    stats_df = pd.DataFrame(result["individual_stats"]).T
    stats_df.columns = ["Expected Return", "Volatility"]
    stats_df["Expected Return"] = stats_df["Expected Return"].apply(lambda x: f"{x:.2%}")
    stats_df["Volatility"] = stats_df["Volatility"].apply(lambda x: f"{x:.2%}")
    st.dataframe(stats_df, use_container_width=True)


def _render_strategy_comparison(strategies: dict):
    rows = []
    for name, p in strategies.items():
        rows.append({"Strategy": name, "Expected Return": f"{p.expected_return:.2%}",
                     "Volatility": f"{p.volatility:.2%}", "Sharpe Ratio": f"{p.sharpe_ratio:.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    weight_data = []
    for name, p in strategies.items():
        for t, w in p.weights.items():
            if w > 0.001:
                weight_data.append({"Strategy": name, "Ticker": t, "Weight": w})
    if weight_data:
        fig = px.bar(pd.DataFrame(weight_data), x="Ticker", y="Weight",
                     color="Strategy", barmode="group", text_auto=".1%")
        fig.update_layout(yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig, use_container_width=True)


def _render_backtest(backtests: dict):
    fig = go.Figure()
    colors = {"Max Sharpe": "red", "Optimal (Max Sharpe)": "red",
              "Equal Weight": "green", "Risk Parity": "blue"}
    for name, bt in backtests.items():
        pv = bt["portfolio_value"]
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values, mode="lines", name=name,
            line=dict(color=colors.get(name, "gray"), width=2),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value ($1 start)",
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for name, bt in backtests.items():
        rows.append({"Strategy": name, "Total Return": f"{bt['total_return']:.2%}",
                     "Ann. Return": f"{bt['annualized_return']:.2%}",
                     "Ann. Vol": f"{bt['annualized_vol']:.2%}",
                     "Max Drawdown": f"{bt['max_drawdown']:.2%}",
                     "Sharpe": f"{bt['sharpe']:.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_export(result: dict, strategies: dict | None, backtests: dict | None):
    st.subheader("Export")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_df = pd.DataFrame([result["optimal"].weights], index=["Weight"]).T
        w_df.index.name = "Ticker"
        st.download_button("Download Weights (CSV)", w_df.to_csv(), "optimal_weights.csv", "text/csv")
    with col2:
        if strategies:
            rows = [{"Strategy": n, "Return": p.expected_return, "Vol": p.volatility,
                     "Sharpe": p.sharpe_ratio, **{f"w_{t}": w for t, w in p.weights.items()}}
                    for n, p in strategies.items()]
            st.download_button("Download Strategies (CSV)", pd.DataFrame(rows).to_csv(index=False),
                               "strategy_comparison.csv", "text/csv")
    with col3:
        if backtests:
            rows = [{"Strategy": n, "Total Return": b["total_return"], "Ann. Return": b["annualized_return"],
                     "Ann. Vol": b["annualized_vol"], "Max DD": b["max_drawdown"], "Sharpe": b["sharpe"]}
                    for n, b in backtests.items()]
            st.download_button("Download Backtest (CSV)", pd.DataFrame(rows).to_csv(index=False),
                               "backtest_results.csv", "text/csv")