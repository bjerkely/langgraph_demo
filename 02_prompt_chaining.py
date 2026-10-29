"""02 - Prompt chaining.

Each LLM call processes the output of the previous one, with a gate between
steps: a trade plan that already defines entry, stop and target is done; an
incomplete one gets a risk section and an invalidation section added.

Run: python 02_prompt_chaining.py
"""

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import (
    ACCOUNT_EQUITY,
    RISK_PCT,
    banner,
    get_llm,
    market_snapshot,
    show_graph,
)

llm = get_llm()


class State(TypedDict):
    symbol: str
    plan: str
    risk_section: str
    final_plan: str


def draft_plan(state: State):
    """First LLM call: draft the trade plan."""
    msg = llm.invoke(
        "Write a short swing-trade plan for "
        f"{state['symbol']}. Use this market data:\n{market_snapshot([state['symbol']])}\n"
        "State the setup, the entry trigger, the stop-loss and the target."
    )
    return {"plan": msg.content}


def check_completeness(state: State):
    """Gate: a usable plan must name an entry, a stop and a target.

    Deterministic on purpose (same spirit as the docs' "?"/"!" check): the point
    is that a *code* decision, not the model, chooses the edge.
    """
    plan = state["plan"].lower()
    has_levels = all(word in plan for word in ("entry", "stop", "target"))
    has_numbers = len(re.findall(r"\d+\.\d+|\d{2,}", state["plan"])) >= 3
    return "Pass" if has_levels and has_numbers else "Fail"


def add_risk_rules(state: State):
    """Second LLM call: size the position and spell out the risk."""
    msg = llm.invoke(
        f"Account: ${ACCOUNT_EQUITY:,.0f}, risking {RISK_PCT:.0%} per trade.\n"
        f"Trade plan:\n{state['plan']}\n\n"
        "Add a risk section: position size in shares, risk per share, total "
        "dollar risk, and the R-multiple to target."
    )
    return {"risk_section": msg.content}


def add_invalidation(state: State):
    """Third LLM call: state what would prove the thesis wrong."""
    msg = llm.invoke(
        f"Trade plan:\n{state['plan']}\n\n"
        f"Risk section:\n{state['risk_section']}\n\n"
        "Add explicit invalidation criteria: what price action or news would "
        "make you exit before the stop is hit."
    )
    return {"final_plan": msg.content}


workflow = StateGraph(State)

workflow.add_node("draft_plan", draft_plan)
workflow.add_node("add_risk_rules", add_risk_rules)
workflow.add_node("add_invalidation", add_invalidation)

workflow.add_edge(START, "draft_plan")
workflow.add_conditional_edges(
    "draft_plan",
    check_completeness,
    {"Fail": "add_risk_rules", "Pass": END},
)
workflow.add_edge("add_risk_rules", "add_invalidation")
workflow.add_edge("add_invalidation", END)

chain = workflow.compile()


def main() -> None:
    banner("Prompt chaining")
    show_graph(chain, "02_prompt_chaining")

    for symbol in ("NVDA", "AAPL"):
        state = chain.invoke({"symbol": symbol})
        took = (
            "draft -> risk -> invalidation"
            if "risk_section" in state
            else "draft -> END"
        )
        print(f"\n### {symbol}  (path: {took})")
        print(f"\nDraft plan:\n{state['plan']}")
        if "risk_section" in state:
            print(f"\n--- --- ---\n\nRisk section:\n{state['risk_section']}")
            print(f"\n--- --- ---\n\nInvalidation:\n{state['final_plan']}")

    # Most drafts name all three levels, so the gate rarely fails. Show what the
    # other branch does with a plan that has no stop, by feeding the node
    # functions directly.
    banner("Gate check: what the Fail branch produces")
    incomplete = {"symbol": "NVDA", "plan": "Buy NVDA if it breaks 180 and ride it."}
    print(f"check_completeness({incomplete['plan']!r}) -> {check_completeness(incomplete)}")
    risk = add_risk_rules(incomplete)
    final = add_invalidation({**incomplete, **risk})
    print(f"\nRisk section:\n{risk['risk_section']}")
    print(f"\nInvalidation:\n{final['final_plan']}")


if __name__ == "__main__":
    main()
