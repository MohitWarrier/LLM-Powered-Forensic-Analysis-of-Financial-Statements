import logging

import streamlit as st
import pandas as pd

from src.features.forensic.service import ForensicService
from src.llm.groq_client import is_llm_available
from src.ui.components import render_trend_chart

logger = logging.getLogger(__name__)


def render_forensic_page(service: ForensicService):
    """Full Streamlit page for forensic analysis."""
    st.title("Forensic Analysis of Financial Statements")
    st.markdown("Analyze a company's financials and detect red flags — powered by LLaMA 3 (Groq)")

    st.subheader("Company Financial Data")

    data_source = st.radio("Choose data source:", ["Use Sample Data", "Upload Your Own Data"])

    df = None
    if data_source == "Use Sample Data":
        df = service.get_sample_data()
        logger.info("Loaded sample forensic data, shape=%s", df.shape)
        st.success("Using sample company data")
        st.download_button(
            label="Download Sample CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="sample_financials.csv",
            mime="text/csv",
            help="Download a sample CSV file with expected format",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload financial data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload a file with financial metrics as rows and years as columns",
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, index_col=0)
                else:
                    df = pd.read_excel(uploaded_file, index_col=0)
                logger.info("Uploaded file '%s', shape=%s", uploaded_file.name, df.shape)
                st.success("File uploaded successfully!")
            except Exception as e:
                logger.error("Error reading uploaded file: %s", e)
                st.error(f"Error reading file: {str(e)}")
                st.stop()
        else:
            st.info("Please upload a financial data file to continue")
            st.stop()

    # Validation
    missing = service.validate_data(df)
    if missing:
        logger.warning("Missing required metrics: %s", missing)
        st.warning(f"Missing required metrics: {', '.join(missing)}")
        st.info("Your data should include these financial metrics as row names.")
        with st.expander("Expected Data Format"):
            from src.features.forensic.service import REQUIRED_METRICS
            sample_format = pd.DataFrame({
                "2022": [800000, 350000, 4000, 800000, 120000, 900000, 1300000],
                "2023": [1000000, 400000, 5000, 1000000, 150000, 1000000, 1500000],
            }, index=REQUIRED_METRICS)
            st.dataframe(sample_format)
        st.stop()

    # Display data
    st.subheader("Financial Data Overview")
    st.dataframe(df, width="stretch")

    # Trend chart
    if df.shape[1] > 1:
        render_trend_chart(df)

    # Session state
    if "forensic_results" not in st.session_state:
        st.session_state["forensic_results"] = None
    if "forensic_llm_output" not in st.session_state:
        st.session_state["forensic_llm_output"] = None

    # Analysis
    if st.button("Analyze Red Flags", type="primary"):
        with st.spinner("Analyzing financial data..."):
            logger.info("Running forensic analysis...")
            st.session_state["forensic_results"] = service.analyze(df)
            st.session_state["forensic_llm_output"] = None
            logger.info("Forensic analysis complete")

    if st.session_state["forensic_results"] is not None:
        results = st.session_state["forensic_results"]
        _render_results(results)
        _render_llm_section(service, results)

    # How to use
    with st.expander("How to Use This Tool"):
        st.markdown("""
        **Step 1**: Choose to use sample data or upload your own financial data

        **Step 2**: If uploading, make sure your file has:
        - Financial metrics as row names (e.g., "Net Income", "Operating Cash Flow")
        - Years as column headers (e.g., "2022", "2023")
        - Numerical values in the cells

        **Step 3**: Click "Analyze Red Flags" to run the detection

        **Step 4**: Review the results and get AI-powered insights

        **Supported Formats**: CSV, Excel (.xlsx, .xls)
        """)


def _render_results(results: dict):
    """Display analysis results with risk scoring."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Detection Results")
        flagged_count = 0
        for metric, details in results.items():
            if "error" in details:
                st.warning(f"**{metric.replace('_', ' ').title()}**: {details['error']}")
            else:
                if details["flagged"]:
                    st.error(f"**{metric.replace('_', ' ').title()}**: {details['value']} — {details['reason']}")
                    flagged_count += 1
                else:
                    st.success(f"**{metric.replace('_', ' ').title()}**: {details['value']} — {details['reason']}")
    with col2:
        total_checks = len([r for r in results.values() if "error" not in r])
        risk_percentage = (flagged_count / total_checks * 100) if total_checks > 0 else 0
        st.metric("Risk Score", f"{risk_percentage:.0f}%")
        st.metric("Red Flags", f"{flagged_count}/{total_checks}")
        if risk_percentage >= 50:
            st.error("High Risk")
        elif risk_percentage >= 25:
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")
        logger.info("Risk score: %.0f%%, flagged: %d/%d", risk_percentage, flagged_count, total_checks)


def _render_llm_section(service: ForensicService, results: dict):
    """LLM explanation section."""
    st.subheader("AI-Powered Analysis")

    if not is_llm_available():
        st.info(
            "LLM features are disabled. To enable, add your Groq API key to a `.env` file "
            "in the project root:\n\n`OPENAI_API_KEY=your_groq_api_key_here`"
        )
        return

    if st.button("Generate LLM Explanation"):
        with st.spinner("Getting AI insights..."):
            try:
                logger.info("Requesting LLM explanation...")
                explanation = service.explain(results)
                st.session_state["forensic_llm_output"] = explanation
                logger.info("LLM explanation received")
            except Exception as e:
                logger.error("LLM error: %s", e)
                st.session_state["forensic_llm_output"] = f"LLM error: {str(e)}"

    if st.session_state["forensic_llm_output"] is not None:
        st.markdown("### LLM Explanation")
        st.write(st.session_state["forensic_llm_output"])
