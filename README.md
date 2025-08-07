# LLM-Powered Forensic Analysis of Financial Statements

This project uses large language models (LLMs) combined with financial forensic accounting techniques to analyze company financials for potential red flags. It provides quantitative checks on key financial metrics and delivers AI-powered explanations to help interpret the results.

## Project Overview

- Users can upload company financial data (CSV or Excel) or use embedded sample data.
- The app runs several forensic checks on the latest financial year (and some tests using multi-year data), including:
  - Cash Conversion Ratio
  - Yield on Cash
  - Contingent Liabilities to Net Worth ratio
  - Debt-to-Equity Ratio
  - Year-over-Year Revenue Growth Rate
- Results highlight flagged metrics with explanations of potential risks.
- An integrated LLM (via Groq client) generates natural language summaries and insights based on the analysis.
- Interactive Streamlit web interface provides data upload, validation, visualization, and analysis.

## Why This Matters

Financial statements often hide warning signs that require detailed forensic analysis. This tool helps users quickly identify such red flags and provides expert-like AI insights, making complex financial data more accessible and understandable.

## How It Works

1. **Data Input**: Upload your financial data in a structured CSV or Excel file, or use the provided sample data.  
2. **Data Validation**: Checks for required financial metrics and displays clear error/warning messages if inputs are incomplete or malformed.  
3. **Financial Forensic Checks**: Runs multiple quantitative red flag analyses on the most recent data (and trends where applicable).  
4. **Trend Visualization**: Shows simple multi-year trends for key financial metrics.  
5. **AI-Powered Explanation**: Upon user request, generates an AI-based narrative explaining the flagged results.  
6. **Results Summary**: Visual indicators of risk levels and a detailed explanation output.

## Current Features

- Upload CSV or Excel files with financial metrics as rows and years as columns.  
- Robust validation with user guidance and formatted error messages.  
- Five key forensic financial red flags calculated.  
- Multi-year financial trend visualization using Plotly.  
- LLM-driven explanation generation with error handling.  
- Downloadable sample CSV file to guide input formatting.

## Target Users

- Finance students and freshers learning forensic accounting and AI applications.  
- Individual investors seeking deeper financial statement insights.  
- Data scientists and developers interested in applying LLMs to financial analysis.

## Getting Started

- Run the app locally with Streamlit:  streamlit run app.py
- Upload your financial statements or use the sample data.  
- Explore analysis results, risk scores, and AI-generated explanations.  
- Use the downloadable sample CSV file to format your own data correctly.

## Planned Next Steps

- Add additional financial checks (e.g., working capital changes, expense pattern irregularities).  
- Integrate automated financial data ingestion via APIs like yfinance.  
- Improve user experience with more flexible input handling and improved error messaging.  
- Prepare the app for deployment and sharing through public hosting.

---

