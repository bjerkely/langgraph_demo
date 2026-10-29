"""01 - LLMs and augmentations.

The two augmentations every workflow/agent is built from:
  * structured output -> turn a rambling trade idea into typed fields
  * tool binding      -> let the model request a position-sizing calculation

Run: python 01_augmentations.py
"""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common import (
    ACCOUNT_EQUITY,
    RISK_PCT,
    banner,
    get_llm,
    shares_for,
    structured,
)


class TradeIntent(BaseModel):
    """A swing-trade idea extracted from a free-text message."""

    symbol: str = Field(description="Ticker the trader is talking about, e.g. NVDA.")
    setup: Literal["breakout", "pullback", "mean_reversion", "trend_continuation"] = Field(
        description="The setup the trader wants to trade."
    )
    holding_period: str | None = Field(
        default=None, description="How long they expect to hold, e.g. '2-3 weeks'."
    )
    notes: str | None = Field(
        default=None, description="Anything else relevant: levels, catalysts, concerns."
    )


@tool
def position_size(
    account_equity: float, risk_pct: float, entry: float, stop: float
) -> str:
    """Size a long swing trade. Always use this tool instead of doing the math.

    Args:
        account_equity: Total account value in dollars.
        risk_pct: Fraction of equity to risk on this trade, e.g. 0.01 for 1%.
        entry: Planned entry price.
        stop: Stop-loss price (below entry for a long).
    """
    shares = shares_for(account_equity, risk_pct, entry, stop)
    risk_per_share = entry - stop
    return (
        f"{shares} shares, ${risk_per_share:.2f} risk per share, "
        f"${shares * risk_per_share:,.2f} total risk "
        f"({shares * entry:,.2f} notional)"
    )


def main() -> None:
    llm = get_llm()

    banner("Structured output: free-text idea -> TradeIntent")
    intent = structured(TradeIntent).invoke(
        "I like NVDA on a pullback to the 20-EMA over the next two weeks, "
        "as long as it holds above 170."
    )
    print(type(intent).__name__, "->")
    print(intent)

    banner("Tool calling: position sizing")
    msg = llm.bind_tools([position_size]).invoke(
        f"I have ${ACCOUNT_EQUITY:,.0f} and I risk {RISK_PCT:.0%} per trade. "
        "Entry 182.50, stop 176.00. How many shares should I buy?"
    )
    print("content:", repr(msg.content))
    print("tool_calls:", msg.tool_calls)
    for call in msg.tool_calls:
        print("tool result:", position_size.invoke(call["args"]))


if __name__ == "__main__":
    main()
