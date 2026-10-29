"""06 - Evaluator-optimizer.

One LLM drafts a trade plan, a risk manager grades it. The grade is checked
against hard rules (stop definition, R-multiple, risk budget) so the loop cannot
be talked into approving a bad plan, and a conditional edge sends rejected
plans back to the drafter with the feedback. This is the first pattern whose
graph has a cycle.

Run: python 06_evaluator_optimizer.py
"""

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common import (
    ACCOUNT_EQUITY,
    RISK_PCT,
    banner,
    get_llm,
    market_snapshot,
    r_multiple,
    shares_for,
    show_graph,
    structured,
)

llm = get_llm()
MAX_ATTEMPTS = 4


class State(TypedDict):
    symbol: str
    plan: dict
    metrics: dict
    verdict: str
    feedback: str
    attempts: int


class TradePlan(BaseModel):
    """A long swing-trade plan."""

    symbol: str = Field(description="Ticker.")
    setup: Literal["breakout", "pullback", "mean_reversion", "trend_continuation"] = Field(
        description="Setup being traded."
    )
    entry: float = Field(description="Entry price for a long.")
    stop: float = Field(description="Stop-loss price, below entry.")
    target: float = Field(description="Target price, above entry.")
    thesis: str = Field(description="Two sentences on why this works.")


class Review(BaseModel):
    """Risk-desk verdict on a trade plan."""

    grade: Literal["approved", "rejected"] = Field(
        description="Decide if the plan is acceptable as written.",
    )
    feedback: str = Field(
        description="If rejected, exactly what to change: stop distance, target, sizing.",
    )


plan_writer = structured(TradePlan)
evaluator = structured(Review)


def llm_call_generator(state: State):
    """LLM drafts (or revises) the trade plan."""
    if state.get("feedback"):
        msg = (
            f"Revise the {state['symbol']} swing-trade plan to satisfy the risk "
            f"desk. Their feedback: {state['feedback']}"
        )
    else:
        msg = f"Draft a long swing-trade plan for {state['symbol']}."

    plan = plan_writer.invoke(
        [
            SystemMessage(
                content=(
                    "You are a disciplined swing trader. Long-only, daily bars. "
                    f"Account ${ACCOUNT_EQUITY:,.0f}, risking {RISK_PCT:.0%} per trade. "
                    "Market data:\n" + market_snapshot([state["symbol"]])
                )
            ),
            HumanMessage(content=msg),
        ]
    )
    return {"plan": plan.model_dump(), "attempts": state.get("attempts", 0) + 1}


def llm_call_evaluator(state: State):
    """Risk manager grades the plan, then hard rules have the final say."""
    plan = state["plan"]
    r = r_multiple(plan["entry"], plan["stop"], plan["target"])
    shares = shares_for(ACCOUNT_EQUITY, RISK_PCT, plan["entry"], plan["stop"])
    risk_dollars = shares * (plan["entry"] - plan["stop"])

    review = evaluator.invoke(
        [
            SystemMessage(
                content=(
                    "You are a risk manager. Approve only if the setup is coherent "
                    "and the stop is beyond an obvious invalidation level rather "
                    "than inside the noise. R-multiple must be at least 2.0."
                )
            ),
            HumanMessage(
                content=(
                    f"Plan: {plan}\n"
                    f"Computed: R-multiple {r:.2f}, {shares} shares, "
                    f"${risk_dollars:,.2f} at risk "
                    f"(budget ${ACCOUNT_EQUITY * RISK_PCT:,.2f})."
                )
            ),
        ]
    )

    # Code, not the model, decides the hard constraints.
    violations = []
    if plan["stop"] >= plan["entry"]:
        violations.append("stop must be below entry")
    if r < 2.0:
        violations.append(f"R-multiple {r:.2f} < 2.0 (widen the target or tighten the stop)")
    if risk_dollars > ACCOUNT_EQUITY * RISK_PCT * 1.05:
        violations.append(f"${risk_dollars:,.2f} at risk exceeds the 1% budget")

    approved = review.grade == "approved" and not violations
    feedback = review.feedback
    if violations:
        feedback = f"{feedback}\nHard-rule violations: {'; '.join(violations)}"

    return {
        "metrics": {
            "r_multiple": round(r, 2),
            "shares": shares,
            "risk_dollars": round(risk_dollars, 2),
            "violations": violations,
        },
        "verdict": "approved" if approved else "rejected",
        "feedback": feedback,
    }


def route_plan(state: State):
    """Approve, revise, or -- after MAX_ATTEMPTS -- escalate to a human."""
    if state["verdict"] == "approved":
        return "Accepted"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "Escalated"
    return "Rejected + Feedback"


optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_plan,
    {  # Name returned by route_plan : Name of node to visit
        "Accepted": END,
        "Escalated": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

optimizer_workflow = optimizer_builder.compile()


def main() -> None:
    banner("Evaluator-optimizer")
    show_graph(optimizer_workflow, "06_evaluator_optimizer")

    # Stream the updates so the cycle is visible: repeated "llm_call_generator"
    # entries are the back-edge doing its job.
    state: dict = {}
    path: list[str] = []
    for update in optimizer_workflow.stream(
        {"symbol": "NVDA"},
        # Guard against an evaluator that never approves.
        config={"recursion_limit": 20},
        stream_mode="updates",
    ):
        for node, delta in update.items():
            path.append(node)
            state.update(delta or {})

    print("path:", " -> ".join(path))
    print("drafts:", path.count("llm_call_generator"))
    print("verdict:", state["verdict"])
    print("metrics:", state["metrics"])
    print("plan:")
    for key, value in state["plan"].items():
        print(f"  {key}: {value}")
    if state["verdict"] != "approved":
        print("escalated to human review, last feedback:\n", state["feedback"])


if __name__ == "__main__":
    main()
