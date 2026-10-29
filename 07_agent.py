"""07 - Agent (tool-calling loop).

Unlike the previous patterns the path is not predetermined: the model decides
which tools to call, and the conditional edge loops until it stops asking.

Run: python 07_agent.py
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from common import (
    ACCOUNT_EQUITY,
    RISK_PCT,
    SAMPLE_MARKET,
    banner,
    get_llm,
    r_multiple,
    shares_for,
    show_graph,
)

llm = get_llm()


@tool
def quote(symbol: str) -> str:
    """Get the last price and ATR(14) for a ticker.

    Args:
        symbol: Ticker, e.g. NVDA.
    """
    symbol = symbol.upper()
    if symbol not in SAMPLE_MARKET:
        return f"unknown ticker {symbol}; known: {', '.join(SAMPLE_MARKET)}"
    last, atr = SAMPLE_MARKET[symbol]
    return f"{symbol}: last {last:.2f}, ATR(14) {atr:.2f}"


@tool
def position_size(account_equity: float, risk_pct: float, entry: float, stop: float) -> str:
    """Size a long position from equity, risk fraction, entry and stop.

    Args:
        account_equity: Total account value in dollars.
        risk_pct: Fraction of equity to risk, e.g. 0.01 for 1%.
        entry: Planned entry price.
        stop: Stop-loss price, below entry.
    """
    shares = shares_for(account_equity, risk_pct, entry, stop)
    risk_per_share = entry - stop
    return (
        f"{shares} shares, ${risk_per_share:.2f} risk per share, "
        f"${shares * risk_per_share:,.2f} total risk"
    )


@tool
def reward_to_risk(entry: float, stop: float, target: float) -> str:
    """Compute the R-multiple (reward/risk) of a long trade.

    Args:
        entry: Planned entry price.
        stop: Stop-loss price, below entry.
        target: Target price, above entry.
    """
    return f"R-multiple {r_multiple(entry, stop, target):.2f}"


tools = [quote, position_size, reward_to_risk]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not."""
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a risk-desk assistant for a swing trader. Use "
                            "the tools for prices, sizing and R-multiple instead of "
                            "computing yourself. Answer in at most 5 lines."
                        )
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState):
    """Performs the tool calls."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        observation = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Continue the loop if the LLM asked for a tool call, otherwise stop."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()


def main() -> None:
    banner("Agent")
    show_graph(agent, "07_agent")

    messages = [
        HumanMessage(
            content=(
                f"I'm long NVDA from 178.40 with a stop at 171.90 and a target of "
                f"191.00. My account is ${ACCOUNT_EQUITY:,.0f} and I risk "
                f"{RISK_PCT:.0%} per trade. Check the quote, size the position, "
                "and tell me the R-multiple."
            )
        )
    ]
    result = agent.invoke({"messages": messages}, config={"recursion_limit": 20})
    for m in result["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
