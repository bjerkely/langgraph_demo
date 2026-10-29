"""Shared helpers for the LangGraph "Workflows and agents" walkthrough.

The examples are swing-trading exercises: the graph mechanics are the same as
the docs, the domain is a $50k account trading US equities on daily/weekly
timeframes.

Everything provider-, environment- or market-specific lives here so the seven
pattern scripts stay as close to the documentation as possible.

Config comes from `.env`:

    OLLAMA_BASE_URL        https://ollama.com  (Ollama Cloud, no local runtime)
    OLLAMA_API_KEY         API key from https://ollama.com/settings/keys
    OLLAMA_MODEL           e.g. gpt-oss:120b-cloud
    OLLAMA_TIMEOUT_SECONDS
    ACCOUNT_EQUITY         optional, defaults to 50000
    RISK_PCT               optional, defaults to 0.01 (1% of equity per trade)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent
GRAPHS_DIR = ROOT / "graphs"

# Values in .env are stripped defensively: the file has been CRLF at times and a
# stray "\r" silently corrupts the base URL / API key.
load_dotenv(ROOT / ".env")


def _env(name: str, default: str | None = None) -> str:
    value = (os.getenv(name) or default or "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {ROOT / '.env'} (see README.md)."
        )
    return value


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    return float(raw) if raw else default


# --------------------------------------------------------------------------- #
# Trading fixtures
#
# Hardcoded, deterministic market data: no broker, no API, no rate limits.
# (ticker: (last price, ATR(14)))
# --------------------------------------------------------------------------- #
SAMPLE_MARKET: dict[str, tuple[float, float]] = {
    "NVDA": (178.40, 5.80),
    "AMD": (142.15, 4.90),
    "AAPL": (231.60, 3.70),
    "MSFT": (438.20, 6.10),
    "SPY": (574.30, 4.20),
    "SMH": (268.90, 5.40),
}

ACCOUNT_EQUITY = _float_env("ACCOUNT_EQUITY", 50_000.0)
RISK_PCT = _float_env("RISK_PCT", 0.01)


def market_snapshot(symbols: list[str] | None = None) -> str:
    """One-line-per-ticker market table, for pasting into prompts."""
    rows = []
    for symbol, (last, atr) in SAMPLE_MARKET.items():
        if symbols and symbol not in symbols:
            continue
        rows.append(f"{symbol}: last {last:.2f}, ATR(14) {atr:.2f}")
    return "\n".join(rows)


def shares_for(equity: float, risk_pct: float, entry: float, stop: float) -> int:
    """Whole shares such that (entry - stop) * shares <= equity * risk_pct."""
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        raise ValueError(f"stop {stop} must be below entry {entry} for a long")
    return math.floor(equity * risk_pct / risk_per_share)


def r_multiple(entry: float, stop: float, target: float) -> float:
    """Reward-to-risk of a long: (target - entry) / (entry - stop)."""
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        raise ValueError(f"stop {stop} must be below entry {entry} for a long")
    return (target - entry) / risk_per_share


def get_llm(**overrides: Any) -> ChatOllama:
    """Return a ChatOllama client pointed at Ollama Cloud.

    `reasoning=False` disables the thinking phase. Models such as gpt-oss spend
    hundreds of tokens "thinking" before answering; for these short tutorial
    nodes that is pure latency and cost.
    """
    api_key = _env("OLLAMA_API_KEY")
    options: dict[str, Any] = {
        "model": _env("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        "base_url": _env("OLLAMA_BASE_URL", "https://ollama.com"),
        "temperature": 0,
        "reasoning": False,
        "client_kwargs": {
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": float(_env("OLLAMA_TIMEOUT_SECONDS", "600")),
        },
    }
    options.update(overrides)
    return ChatOllama(**options)


def structured(schema: type[BaseModel], **overrides: Any) -> RunnableLambda:
    """Drop-in replacement for `llm.with_structured_output(schema)`.

    Two Ollama-specific problems make the stock call unreliable here:

    1. `with_structured_output` defaults to method="json_schema", which relies on
       Ollama's `format` parameter. The cloud endpoint does not enforce it -- the
       model answers in prose or invents its own JSON shape.
    2. Even the working function-calling path returns `None` (silently, no error)
       when the model decides to answer the question instead of calling the tool.

    So we force the tool call with a system message, verify a tool call actually
    came back, re-prompt once if it did not, and validate the arguments against
    the schema. Scripts still read `router = structured(Route)`.
    """
    llm = get_llm(**overrides)
    tool_name = convert_to_openai_tool(schema)["function"]["name"]
    bound = llm.bind_tools([schema])
    system = SystemMessage(
        content=(
            f"You are a data-extraction function. Respond by calling the "
            f"`{tool_name}` tool exactly once with arguments matching its "
            "schema. Never answer in prose."
        )
    )

    def _invoke(input_: Any, config: Any = None) -> BaseModel:
        if isinstance(input_, str):
            messages: list[BaseMessage] = [system, HumanMessage(content=input_)]
        elif isinstance(input_, BaseMessage):
            messages = [system, input_]
        else:
            messages = [system, *input_]

        last_error: Exception | None = None
        for _ in range(3):
            ai = bound.invoke(messages, config)
            if ai.tool_calls:
                args = ai.tool_calls[0]["args"]
                try:
                    return schema(**args)
                except ValidationError as exc:
                    # The model called the tool but with the wrong shape, e.g. a
                    # list of strings where a list of objects was required.
                    last_error = exc
                    messages = messages + [
                        HumanMessage(
                            content=(
                                f"That tool call did not match the schema: {exc}\n"
                                f"Call `{tool_name}` again with arguments matching "
                                "the schema exactly."
                            )
                        )
                    ]
                    continue
            messages = messages + [
                ai,
                HumanMessage(content=f"Call the `{tool_name}` tool now."),
            ]

        raise RuntimeError(
            f"{llm.model} never produced a valid `{tool_name}` call. "
            f"Last error: {last_error}. Last reply: {ai.content!r}"
        )

    return RunnableLambda(_invoke)


def show_graph(graph: Any, name: str) -> str:
    """Write the graph's Mermaid source to graphs/<name>.mmd and return it.

    Replaces the tutorial's `display(Image(graph.get_graph().draw_mermaid_png()))`,
    which needs pyppeteer/network and only makes sense inside a notebook.
    """
    GRAPHS_DIR.mkdir(exist_ok=True)
    mermaid = graph.get_graph().draw_mermaid()
    path = GRAPHS_DIR / f"{name}.mmd"
    path.write_text(mermaid, encoding="utf-8")
    print(f"[graph] {path.relative_to(ROOT)}")
    return mermaid


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
