import json
import logging

import pandas as pd

from config import settings
from src.core.interfaces import BaseService, DataProvider
from src.features.portfolio.optimizer import MeanVarianceOptimizer
from src.features.quant_analyst.tools import TOOL_DEFINITIONS, TOOL_REGISTRY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI Quant Analyst. You help users understand their portfolio's
risk, performance, and allocation.

You have access to computational tools that analyze portfolio data. ALWAYS use the
appropriate tools to gather data before answering quantitative questions. Never guess
numbers — call the tools first.

When answering:
- Be precise with numbers (cite specific metrics from tool results)
- Explain financial concepts in accessible language
- Highlight actionable insights
- Keep answers concise but informative"""

MAX_TOOL_ITERATIONS = 5


class QuantAnalystService(BaseService):
    """AI-powered portfolio analysis with LLM tool-use."""

    def __init__(self, data_provider: DataProvider):
        self._data_provider = data_provider
        self._yfinance_provider = None
        self._optimizer = MeanVarianceOptimizer()
        logger.info("QuantAnalystService initialized")

    def get_name(self) -> str:
        return "AI Quant Analyst"

    def get_description(self) -> str:
        return "Ask questions about your portfolio in natural language"

    def get_sample_data(self) -> pd.DataFrame:
        return self._data_provider.get_portfolio_data()

    def get_live_data(self, tickers: list[str], period: str = "2y") -> pd.DataFrame:
        if self._yfinance_provider is None:
            from src.data.yfinance_provider import YFinanceProvider
            self._yfinance_provider = YFinanceProvider(tickers, period)
        else:
            self._yfinance_provider.set_tickers(tickers)
            self._yfinance_provider.set_period(period)
        return self._yfinance_provider.get_portfolio_data()

    def run_chat(self, user_message: str, chat_history: list[dict],
                 prices: pd.DataFrame, weights: dict[str, float],
                 risk_free_rate: float = 0.02) -> tuple[str, list[dict]]:
        """
        Run the LLM tool-use chat loop.

        Args:
            user_message: The user's question
            chat_history: Previous messages in OpenAI format
            prices: Loaded price data
            weights: Current portfolio weights
            risk_free_rate: Risk-free rate

        Returns:
            (assistant_response, tool_calls_made)
            tool_calls_made is a list of {"tool": name, "result_summary": str}
        """
        from src.llm.groq_client import get_client

        client = get_client()
        context = {
            "prices": prices,
            "weights": weights,
            "optimizer": self._optimizer,
            "risk_free_rate": risk_free_rate,
        }

        # Build message list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            logger.info("Chat iteration %d, sending %d messages", iteration + 1, len(messages))

            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            msg = choice.message

            # Add assistant message to history
            messages.append(msg.model_dump())

            if not msg.tool_calls:
                # LLM produced a final text response
                logger.info("Chat complete after %d iterations, %d tool calls",
                            iteration + 1, len(tool_calls_made))
                return msg.content or "I couldn't generate a response.", tool_calls_made

            # Execute tool calls
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("Executing tool: %s(%s)", fn_name, fn_args)

                if fn_name not in TOOL_REGISTRY:
                    result = {"error": f"Unknown tool: {fn_name}"}
                else:
                    try:
                        result = TOOL_REGISTRY[fn_name](context=context, **fn_args)
                    except Exception as e:
                        logger.error("Tool %s failed: %s", fn_name, e)
                        result = {"error": str(e)}

                tool_calls_made.append({"tool": fn_name, "args": fn_args})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

        # Hit max iterations
        logger.warning("Hit max tool iterations (%d)", MAX_TOOL_ITERATIONS)
        return msg.content or "Analysis incomplete — too many tool calls.", tool_calls_made
