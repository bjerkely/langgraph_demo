"""05 - Orchestrator-worker.

The orchestrator plans a weekly swing-trading plan, the `Send` API spawns one
worker per section, and each worker writes its section into a shared, additive
state key. A final node joins the sections.

The number of workers is decided at runtime by the plan, which is what makes
this different from plain parallelization.

Run: python 05_orchestrator_worker.py
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common import (
    ACCOUNT_EQUITY,
    RISK_PCT,
    banner,
    get_llm,
    market_snapshot,
    show_graph,
    structured,
)

llm = get_llm()


class Section(BaseModel):
    """One section of the trading plan."""

    name: str = Field(description="Name for this section of the plan.")
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section."
    )


class Sections(BaseModel):
    """The full plan for the weekly review."""

    sections: list[Section] = Field(
        description="Sections of the plan, each an object with a name and a description."
    )


planner = structured(Sections)


class State(TypedDict):
    topic: str  # What the plan is for
    sections: list[Section]  # List of sections
    completed_sections: Annotated[list, operator.add]  # Workers write here in parallel
    final_report: str  # Joined plan


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


def orchestrator(state: State):
    """Orchestrator that generates a plan for the weekly review."""
    report_sections = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Generate a plan for a weekly swing-trading review. "
                    f"Account ${ACCOUNT_EQUITY:,.0f}, risking {RISK_PCT:.0%} per trade. "
                    "Watchlist:\n" + market_snapshot()
                )
            ),
            HumanMessage(content=f"Here is the goal: {state['topic']}"),
        ]
    )
    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """Worker writes a section of the plan."""
    section = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Write one section of a swing-trading plan following the "
                    "provided name and description. Include no preamble. Use "
                    "markdown.\n\n"
                    f"Account ${ACCOUNT_EQUITY:,.0f}, risking {RISK_PCT:.0%} per "
                    "trade. Use ONLY these tickers and prices -- do not invent "
                    f"quotes:\n{market_snapshot()}"
                )
            ),
            HumanMessage(
                content=(
                    f"Section name: {state['section'].name}. "
                    f"Description: {state['section'].description}"
                )
            ),
        ]
    )
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize the full plan from sections."""
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}


def assign_workers(state: State):
    """Assign a worker to each section in the plan."""
    # Kick off section writing in parallel via the Send() API.
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()


def main() -> None:
    banner("Orchestrator-worker")
    show_graph(orchestrator_worker, "05_orchestrator_worker")

    state = orchestrator_worker.invoke(
        {"topic": "Weekly swing-trading plan for a $50k account trading US equities"}
    )
    print(f"sections planned: {[s.name for s in state['sections']]}\n")
    print(state["final_report"])


if __name__ == "__main__":
    main()
