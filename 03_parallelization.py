"""03 - Parallelization.

Three independent checks on the same candidate run at once -- technicals,
catalyst and risk -- then an aggregator turns them into one pre-trade brief
with a go/no-go verdict.

LangGraph executes every node that is ready, so fan-out is just "add an edge
from START to each node" and fan-in is "add an edge from each node to the next".

Run: python 03_parallelization.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import ACCOUNT_EQUITY, RISK_PCT, banner, get_llm, market_snapshot, show_graph

llm = get_llm()


class State(TypedDict):
    symbol: str
    technicals: str
    catalyst: str
    risk: str
    brief: str


def call_llm_1(state: State):
    """Technical analyst: trend, location, key levels."""
    msg = llm.invoke(
        f"Act as a technical analyst. Candidate: {state['symbol']}.\n"
        f"{market_snapshot([state['symbol']])}\n"
        "In 6 lines: trend vs the 20/50-day EMA, whether price is extended or "
        "coiled, the nearest support and resistance, and the setup you would take."
    )
    return {"technicals": msg.content}


def call_llm_2(state: State):
    """Catalyst analyst: scheduled events and headline risk."""
    msg = llm.invoke(
        f"Act as an event/catalyst analyst for {state['symbol']}. "
        "In 6 lines: what scheduled events matter over the next 4 weeks "
        "(earnings, guidance, macro prints, index rebalances), what the options "
        "market usually does into them, and whether that argues for waiting."
    )
    return {"catalyst": msg.content}


def call_llm_3(state: State):
    """Risk analyst: stop distance and position size."""
    msg = llm.invoke(
        f"Act as a risk manager sizing a long in {state['symbol']}.\n"
        f"{market_snapshot([state['symbol']])}\n"
        f"Account ${ACCOUNT_EQUITY:,.0f}, risking {RISK_PCT:.0%} per trade. "
        "In 6 lines: a sensible ATR-based stop distance, the implied share "
        "count, the dollar risk, and the biggest way this trade loses money."
    )
    return {"risk": msg.content}


def aggregator(state: State):
    """Combine the three independent checks into one pre-trade brief."""
    brief = f"PRE-TRADE BRIEF - {state['symbol']}\n\n"
    brief += f"TECHNICALS:\n{state['technicals']}\n\n"
    brief += f"CATALYST:\n{state['catalyst']}\n\n"
    brief += f"RISK:\n{state['risk']}"
    return {"brief": brief}


parallel_builder = StateGraph(State)

parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()


def main() -> None:
    banner("Parallelization")
    show_graph(parallel_workflow, "03_parallelization")

    state = parallel_workflow.invoke({"symbol": "NVDA"})
    print(state["brief"])


if __name__ == "__main__":
    main()
