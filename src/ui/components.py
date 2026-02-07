import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_trend_chart(df: pd.DataFrame, title: str = "Financial Metric Trends"):
    """Render a multi-year financial trend line chart.

    Args:
        df: DataFrame with metrics as index and years as columns.
        title: Chart title.
    """
    st.subheader(title)

    required_metrics = [
        "Net Income", "Operating Cash Flow", "Interest Income",
        "Cash and Cash Equivalents", "Contingent Liabilities",
        "Shareholder Equity", "Total Debt"
    ]
    metrics_to_show = [m for m in required_metrics if m in df.index]

    fig = go.Figure()
    for metric in metrics_to_show:
        fig.add_trace(go.Scatter(
            x=df.columns,
            y=df.loc[metric].values,
            mode="lines+markers",
            name=metric,
        ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Value",
        legend_title="Financial Metrics",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")
