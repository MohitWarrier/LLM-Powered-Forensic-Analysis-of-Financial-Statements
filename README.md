# LLM-Powered Forensic Analysis of Financial Statements

This project applies large language models (LLMs) and forensic accounting principles to detect red flags in public company financials. It combines traditional quantitative checks with natural language analysis of company disclosures to identify signs of earnings manipulation, weak financial health, or misleading narratives.

## Project Goals

- Identify financial red flags using well-established forensic metrics such as:
  - Cash Conversion Ratio
  - Yield on Cash
  - ROCE/ROIC
  - Interest Coverage Ratio
  - Growth in receivables or inventory relative to sales
  - Contingent liabilities relative to net worth

- Analyze the language used in annual reports (e.g., MD&A sections, footnotes) to spot vague, evasive, or overly optimistic phrasing often associated with accounting irregularities.

- Provide clear, structured summaries of each company’s financial health and any concerns identified.

## Why This Matters

Companies frequently present a cleaner picture than reality through aggressive accounting or selective disclosure. Most investors either don’t have time to do deep forensic work or aren’t trained to spot these patterns. This tool helps bridge that gap.

The goal isn’t just to process data — it’s to extract insight: what’s really going on with a business, and whether the reported numbers can be trusted.

## How It Works

1. **Data extraction**: Financial statements and notes are collected from public filings.
2. **Quantitative analysis**: A rules engine checks for known red flags using key ratios and trends.
3. **LLM-based review**: A language model reads key sections of the reports, identifies risky language, and provides context or explanation.
4. **Output**: The findings are combined into a summary report for each company.

## Target Audience

This project is intended for:
- Individual investors who want to go beyond surface-level metrics
- Analysts looking for a faster way to screen companies for risk
- Developers interested in applied LLM use cases in finance

## Current Status

Active development. Planned features include:
- Parsing PDFs and 10-Ks directly
- Interactive web interface
- Support for batch screening of multiple companies
- Risk scoring system based on combined quant and language outputs
