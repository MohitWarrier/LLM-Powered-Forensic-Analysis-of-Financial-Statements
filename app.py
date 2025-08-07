import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.analysis.red_flags import analyze_company
from src.llm.formatter import prompt_builder
from src.llm.groq_client import chat_with_groq
import io

st.set_page_config(page_title="Financial Forensic Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 LLM-Powered Forensic Analysis of Financial Statements")
st.markdown("Analyze a company's financials and detect red flags — powered by LLaMA 3 (Groq)")

st.subheader("📊 Company Financial Data")

data_source = st.radio("Choose data source:", ["Use Sample Data", "Upload Your Own Data"])

# Sample data for download
sample_data = {
    "2022": {
        "Revenue": 2000000,
        "Net Income": 800000,
        "Operating Cash Flow": 350000,
        "Interest Income": 4000,
        "Cash and Cash Equivalents": 800000,
        "Contingent Liabilities": 120000,
        "Shareholder Equity": 900000,
        "Total Debt": 1300000
    },
    "2023": {
        "Revenue": 2600000,  # 30% growth, not flagged
        "Net Income": 1000000,
        "Operating Cash Flow": 400000,
        "Interest Income": 5000,
        "Cash and Cash Equivalents": 1000000,
        "Contingent Liabilities": 150000,
        "Shareholder Equity": 1000000,
        "Total Debt": 1500000
    }
}
sample_df = pd.DataFrame(sample_data)

sample_df = pd.DataFrame(sample_data)

# Download button for sample CSV
st.download_button(
    label="Download Sample CSV",
    data=sample_df.to_csv().encode('utf-8'),
    file_name="sample_financials.csv",
    mime='text/csv',
    help="Download a sample CSV file with expected format"
)

df = None
if data_source == "Use Sample Data":
    df = sample_df
    st.success("✅ Using sample company data")
else:
    uploaded_file = st.file_uploader("Upload financial data (CSV or Excel)", type=['csv', 'xlsx', 'xls'],
                                    help="Upload a file with financial metrics as rows and years as columns")
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, index_col=0)
            else:
                df = pd.read_excel(uploaded_file, index_col=0)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.stop()
    else:
        st.info("👆 Please upload a financial data file to continue")
        st.stop()

required_metrics = [
    "Net Income", "Operating Cash Flow", "Interest Income", "Cash and Cash Equivalents",
    "Contingent Liabilities", "Shareholder Equity", "Total Debt"
]
if df is not None:
    missing_metrics = [metric for metric in required_metrics if metric not in df.index]
    if missing_metrics:
        st.warning(f"⚠️ Missing required metrics: {', '.join(missing_metrics)}")
        st.info("Your data should include these financial metrics as row names.")
        with st.expander("📋 Expected Data Format"):
            sample_format = pd.DataFrame({
                "2022": [800000, 350000, 4000, 800000, 120000, 900000, 1300000],
                "2023": [1000000, 400000, 5000, 1000000, 150000, 1000000, 1500000]
            }, index=required_metrics)
            st.dataframe(sample_format)
        st.stop()

st.subheader("📈 Financial Data Overview")
st.dataframe(df, use_container_width=True)

# Multi-year trend visualization
if df is not None and df.shape[1] > 1:
    st.subheader("📈 Financial Metric Trends")
    metrics_to_show = [m for m in required_metrics if m in df.index]
    fig = go.Figure()
    for metric in metrics_to_show:
        fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric].values,
                                 mode='lines+markers', name=metric))
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Value',
        legend_title='Financial Metrics',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = None
if "llm_output" not in st.session_state:
    st.session_state["llm_output"] = None

if st.button("🔍 Analyze Red Flags", type="primary"):
    with st.spinner("Analyzing financial data..."):
        st.session_state["results"] = analyze_company(df)
        st.session_state["llm_output"] = None  # Reset LLM output

if st.session_state["results"] is not None:
    results = st.session_state["results"]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Detection Results")
        flagged_count = 0
        for metric, details in results.items():
            if "error" in details:
                st.warning(f"⚠️ **{metric.replace('_', ' ').title()}**: {details['error']}")
            else:
                if details["flagged"]:
                    st.error(f"❌ **{metric.replace('_', ' ').title()}**: {details['value']} → {details['reason']}")
                    flagged_count += 1
                else:
                    st.success(f"✅ **{metric.replace('_', ' ').title()}**: {details['value']} → {details['reason']}")
    with col2:
        total_checks = len([r for r in results.values() if "error" not in r])
        risk_percentage = (flagged_count / total_checks * 100) if total_checks > 0 else 0
        st.metric("🎯 Risk Score", f"{risk_percentage:.0f}%")
        st.metric("🚩 Red Flags", f"{flagged_count}/{total_checks}")
        if risk_percentage >= 50:
            st.error("🚨 High Risk")
        elif risk_percentage >= 25:
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Low Risk")

    st.subheader("🧠 AI-Powered Analysis")
    if st.button("Generate LLM Explanation"):
        with st.spinner("Getting AI insights..."):
            try:
                prompt = prompt_builder(results)
                explanation = chat_with_groq(prompt)
                st.session_state["llm_output"] = explanation
            except Exception as e:
                st.session_state["llm_output"] = f"❌ LLM error: {str(e)}"

    if st.session_state["llm_output"] is not None:
        st.markdown("### 💬 LLM Explanation")
        st.write(st.session_state["llm_output"])

with st.expander("📖 How to Use This Tool"):
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
