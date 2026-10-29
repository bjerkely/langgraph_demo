"""04 - Routing.

The first LLM call classifies the request with structured output, and a
conditional edge sends it to the matching desk: screening, sizing, or journal
review.

Run: python 04_routing.py
"""

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common import (
    ACCOUNT_EQUITY,
    RISK_PCT,
    SAMPLE_MARKET,
    banner,
    get_llm,
    market_snapshot,
    show_graph,
    structured,
)

llm = get_llm()


class Route(BaseModel):
    """Pick the desk that should handle this request."""

    step: Literal["screener", "risk", "journal"] = Field(
        description=(
            "screener = find/qualify trade setups; "
            "risk = position sizing, stops, exposure; "
            "journal = review a past trade or mistake"
        )
    )


router = structured(Route)


class State(TypedDict):
    input: str
    decision: str
    output: str


def screener(state: State):
    """Find candidates that match the requested setup."""
    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a swing-trading screener. Use only the watchlist "
                    f"below.\n{market_snapshot()}"
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"output": msg.content}


def risk_desk(state: State):
    """Answer sizing / stop questions with real arithmetic."""
    symbol = "NVDA" if "NVDA" in state["input"].upper() else "AAPL"
    last, atr = SAMPLE_MARKET[symbol]
    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are the risk desk. Account ${ACCOUNT_EQUITY:,.0f}, "
                    f"risking {RISK_PCT:.0%} per trade. {symbol} last {last:.2f}, "
                    f"ATR(14) {atr:.2f}. A common stop is 1.5x ATR below entry. "
                    "Show the arithmetic: stop price, risk per share, share count, "
                    "and dollar risk."
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"output": msg.content}


def journal(state: State):
    """Review a closed trade and extract the lesson."""
    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a trading coach reviewing a closed swing trade. "
                    "Separate bad luck from bad process, name the single process "
                    "error, and give one rule to add to the trading plan."
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"output": msg.content}


def llm_call_router(state: State):
    """Route the input to the appropriate desk."""
    decision = router.invoke(
        [
            SystemMessage(
                content=(
                    "Route the request to screener, risk, or journal based on "
                    "what the trader is asking for."
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.step}


def route_decision(state: State):
    if state["decision"] == "screener":
        return "screener"
    elif state["decision"] == "risk":
        return "risk_desk"
    elif state["decision"] == "journal":
        return "journal"


router_builder = StateGraph(State)

router_builder.add_node("screener", screener)
router_builder.add_node("risk_desk", risk_desk)
router_builder.add_node("journal", journal)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of node to visit
        "screener": "screener",
        "risk_desk": "risk_desk",
        "journal": "journal",
    },
)
router_builder.add_edge("screener", END)
router_builder.add_edge("risk_desk", END)
router_builder.add_edge("journal", END)

router_workflow = router_builder.compile()


def main() -> None:
    banner("Routing")
    show_graph(router_workflow, "04_routing")

    requests = [
        "Find me pullback setups in semis",
        "How many NVDA shares do I buy if I enter at 182.50 with a stop at 176.00?",
        "Review my AMD trade: I got stopped out two days before it reversed.",
    ]
    for request in requests:
        state = router_workflow.invoke({"input": request})
        print(f"\n### request: {request}")
        print(f"decision: {state['decision']}")
        print(f"output:\n{state['output']}")


if __name__ == "__main__":
    main()
