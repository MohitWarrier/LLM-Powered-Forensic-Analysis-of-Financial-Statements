import logging

import pandas as pd

from src.core.interfaces import AnalyzerInterface

logger = logging.getLogger(__name__)


class ForensicAnalyzer(AnalyzerInterface):
    """Five forensic financial checks. Preserves all logic from the original red_flags.py."""

    def analyze(self, df: pd.DataFrame) -> dict:
        results = {}

        logger.info("Running forensic analysis on DataFrame shape=%s, columns=%s", df.shape, list(df.columns))

        if df.empty:
            return {"error": "Empty dataframe provided"}

        latest_year = df.columns[-1]

        # 1. Cash Conversion Ratio
        try:
            net_income = df.loc["Net Income", latest_year]
            op_cash_flow = df.loc["Operating Cash Flow", latest_year]

            if net_income == 0:
                results["cash_conversion_ratio"] = {
                    "value": "N/A",
                    "flagged": True,
                    "reason": "Cannot calculate with zero net income"
                }
            else:
                ccr = op_cash_flow / net_income
                results["cash_conversion_ratio"] = {
                    "value": float(round(ccr, 2)),
                    "flagged": ccr < 0.8,
                    "reason": "Low CCR may indicate earnings quality issues" if ccr < 0.8 else "Healthy cash conversion"
                }
        except KeyError as e:
            results["cash_conversion_ratio"] = {"error": f"Missing data: {str(e)}"}
        except Exception as e:
            results["cash_conversion_ratio"] = {"error": f"Calculation error: {str(e)}"}

        # 2. Yield on Cash
        try:
            interest_income = df.loc["Interest Income", latest_year]
            cash = df.loc["Cash and Cash Equivalents", latest_year]

            if cash == 0:
                results["yield_on_cash"] = {
                    "value": "N/A",
                    "flagged": True,
                    "reason": "No cash reported"
                }
            else:
                yield_on_cash = interest_income / cash
                results["yield_on_cash"] = {
                    "value": float(round(yield_on_cash, 4)),
                    "flagged": yield_on_cash < 0.01,
                    "reason": "Cash earning very low returns" if yield_on_cash < 0.01 else "Reasonable return on cash"
                }
        except KeyError as e:
            results["yield_on_cash"] = {"error": f"Missing data: {str(e)}"}
        except Exception as e:
            results["yield_on_cash"] = {"error": f"Calculation error: {str(e)}"}

        # 3. Contingent Liabilities to Net Worth
        try:
            cont_liab = df.loc["Contingent Liabilities", latest_year]
            equity = df.loc["Shareholder Equity", latest_year]

            if equity == 0:
                results["contingent_liability_ratio"] = {
                    "value": "N/A",
                    "flagged": True,
                    "reason": "Zero or negative equity"
                }
            else:
                cl_ratio = cont_liab / equity
                results["contingent_liability_ratio"] = {
                    "value": float(round(cl_ratio, 2)),
                    "flagged": cl_ratio > 0.1,
                    "reason": "High off-balance sheet risk" if cl_ratio > 0.1 else "Acceptable contingent liability level"
                }
        except KeyError as e:
            results["contingent_liability_ratio"] = {"error": f"Missing data: {str(e)}"}
        except Exception as e:
            results["contingent_liability_ratio"] = {"error": f"Calculation error: {str(e)}"}

        # 4. Debt-to-Equity Ratio
        try:
            total_debt = df.loc['Total Debt', latest_year]
            equity = df.loc['Shareholder Equity', latest_year]

            if equity == 0:
                debt_equity_ratio = 'N/A'
                flagged = True
                reason = 'Equity is zero, cannot calculate'
            else:
                debt_equity_ratio = total_debt / equity
                flagged = debt_equity_ratio > 2
                reason = 'High leverage risk' if flagged else 'Leverage acceptable'

            results['debt_to_equity_ratio'] = {
                'value': float(round(debt_equity_ratio, 2)) if debt_equity_ratio != 'N/A' else 'N/A',
                'flagged': flagged,
                'reason': reason
            }
        except KeyError:
            results['debt_to_equity_ratio'] = {'error': 'Total Debt information missing'}
        except Exception as e:
            results['debt_to_equity_ratio'] = {'error': f'Calculation error: {str(e)}'}

        # 5. Revenue Growth Rate (YoY)
        try:
            if "Revenue" in df.index and len(df.columns) > 1:
                latest = df.columns[-1]
                previous = df.columns[-2]

                rev_latest = df.loc["Revenue", latest]
                rev_previous = df.loc["Revenue", previous]

                if rev_previous == 0:
                    growth = 'N/A'
                    flagged = True
                    reason = "Cannot calculate growth from zero in previous year"
                else:
                    growth = (rev_latest - rev_previous) / rev_previous
                    flagged = growth > 0.5 or growth < 0
                    reason = "Unusual revenue growth" if flagged else "Normal revenue growth"

                results["revenue_growth_rate"] = {
                    "value": round(growth * 100, 2) if growth != 'N/A' else 'N/A',
                    "flagged": flagged,
                    "reason": reason
                }
            else:
                results["revenue_growth_rate"] = {"error": "Revenue data missing or insufficient years"}
        except Exception as e:
            results["revenue_growth_rate"] = {"error": f"Calculation error: {str(e)}"}

        flagged = [k for k, v in results.items() if v.get("flagged")]
        errors = [k for k, v in results.items() if "error" in v]
        logger.info("Analysis complete: %d checks, %d flagged, %d errors", len(results), len(flagged), len(errors))
        if flagged:
            logger.info("Flagged metrics: %s", flagged)

        return results
