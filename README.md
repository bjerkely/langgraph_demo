# LangGraph: Workflows and agents

A runnable walkthrough of the LangGraph guide
[**Workflows and agents**](https://docs.langchain.com/oss/python/langgraph/workflows-agents),
implemented with the **Graph API** and running against **Ollama Cloud**
(`https://ollama.com`, no local runtime, no model downloads).

**New to LLMs or LangGraph?** Start with [TUTORIAL.md](TUTORIAL.md) — a
from-scratch walkthrough of the concepts, the seven patterns, and the pitfalls.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`.env` (already present, `chmod 600`):

```ini
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=...            # https://ollama.com/settings/keys
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_TIMEOUT_SECONDS=600

# optional trading fixtures
ACCOUNT_EQUITY=50000
RISK_PCT=0.01
```

Swap `OLLAMA_MODEL` to any cloud model that supports tools
(`deepseek-v3.1:671b-cloud`, `qwen3-coder:480b-cloud`, `kimi-k2:1t-cloud`, …).

## The domain: swing trading, $50k account

Every exercise is a step in one recurring workflow: turn a trade idea into a
sized, risk-checked plan.

| Term | Meaning here |
|---|---|
| **Setup** | `breakout`, `pullback`, `mean_reversion`, `trend_continuation` |
| **Entry / stop / target** | The three prices a usable plan must state |
| **R-multiple** | `(target - entry) / (entry - stop)`; the risk desk requires ≥ 2.0 |
| **Position size** | `floor(equity × risk_pct / (entry - stop))` shares |
| **ATR(14)** | Average true range; stops are placed ~1.5×ATR away so noise doesn't hit them |
| **Invalidation** | What proves the thesis wrong *before* the stop is reached |

Market data is a hardcoded table in `common.py` (`NVDA, AMD, AAPL, MSFT, SPY,
SMH`), so runs are deterministic, offline, and rate-limit free.

## Run

```bash
.venv/bin/python 01_augmentations.py      # or one at a time, in any order
.venv/bin/python 08_event_log.py          # exercise: event log -> incident report
./run_all.sh                              # the seven patterns, logs to logs/
```

Each script prints its output and writes a Mermaid diagram to `graphs/*.mmd`
(render with `npx -y @mermaid-js/mermaid-cli -i graphs/07_agent.mmd -o 07_agent.png`,
or paste into mermaid.live).

## The seven patterns

| # | File | Pattern | What makes it different | Shape of the graph |
|---|------|---------|--------------------------|--------------------|
| 01 | `01_augmentations.py` | LLM augmentations | free-text idea → typed `TradeIntent`; `position_size` tool; no graph yet | — |
| 02 | `02_prompt_chaining.py` | Prompt chaining | deterministic **completeness gate**: a plan missing entry/stop/target gets extra sections | `START → draft → (Pass → END │ Fail → risk → invalidation)` |
| 03 | `03_parallelization.py` | Parallelization | technicals, catalyst and risk analysts run **concurrently**, then aggregate | `START → {3 analysts} → aggregator → END` |
| 04 | `04_routing.py` | Routing | structured-output classifier picks a desk: **screener / risk / journal** | `START → router → {screener │ risk │ journal} → END` |
| 05 | `05_orchestrator_worker.py` | Orchestrator-worker | `Send` spawns **one worker per planned section**, unknown at build time; results merge via `Annotated[list, operator.add]` | `START → orchestrator → (Send × N) worker → synthesizer → END` |
| 06 | `06_evaluator_optimizer.py` | Evaluator-optimizer | the only **cycle**: drafter ⇄ risk manager until the plan passes the hard rules | `drafter ⇄ risk_manager`, exit on `approved` |
| 07 | `07_agent.py` | Agent | no fixed path: the model chains `quote → position_size → reward_to_risk` itself | `llm_call ⇄ tool_node`, exit when no `tool_calls` |

Rules of thumb: **predefined order → workflow** (02–06); **the model picks the
path → agent** (07). Reach for 05 when the number of subtasks is only known at
runtime (the planner usually asks for 5–7 sections), and 06 when there is a
success criterion you can grade — here, stop defined, R ≥ 2, size inside the 1%
budget.

## Exercise: event log → incident report

`08_event_log.py` applies the same structured-output idea to a long,
custom-format machine log, where no single line contains the answer:

```
parse (your adapter) -> aggregate (Python) -> label + correlate (2 LLM calls)
                     -> verify + fill in (Python)
```

- `data/litho_events.log` is a synthetic 600-line log from a lithography tool:
  12 subsystems, one cross-subsystem fault chain, red herrings and malformed
  lines.
- Aggregation collapses ~600 events to ~23 rows *before* prompting, so the run
  costs two calls no matter how long the log is.
- The model returns event ids, labels and causal links. Python owns every count,
  timestamp and line number, and checks that every cited id exists.
- Format is config, not code: `--format bracket|jsonl|kv|auto|spec.json`.
  `formats/example_custom.json` is a template, and
  `data/generate_litho_events.py` can make a different log.

```bash
.venv/bin/python 08_event_log.py --dry-run     # parse + aggregate, no LLM
.venv/bin/python 08_event_log.py --format auto
.venv/bin/python 08_event_log.py mytool.log --out events.jsonl
```

TUTORIAL.md Part 8 has the exercise tiers: write a spec for your own log, break
the verifier, and wrap the pipeline in a graph.

## Helpers (`common.py`)

| Function | Why it exists |
|---|---|
| `get_llm()` | `ChatOllama` wired to `OLLAMA_BASE_URL` + bearer auth, `reasoning=False` |
| `structured(schema)` | `with_structured_output` replacement that survives Ollama Cloud (see below) |
| `show_graph(graph, name)` | writes `graphs/<name>.mmd` — replaces `display(Image(...draw_mermaid_png()))` |
| `banner(title)` | section headers in the console output |
| `SAMPLE_MARKET`, `market_snapshot()` | deterministic prices/ATR pasted into prompts; no API, no rate limits |
| `shares_for()`, `r_multiple()` | the two formulas used by the sizing tool, the risk desk and the agent |

## Four things that differ from the guide

These are not style preferences; each was verified against this endpoint.

1. **`base_url`** — the guide's local Ollama assumption does not hold here.
   `http://localhost:11434` is dead on this machine, so everything points at
   `https://ollama.com` with an `Authorization: Bearer <key>` header in
   `client_kwargs`. ChatOllama's `base_url` alone carries no auth.

2. **Structured output needs `method="function_calling"`.**
   `with_structured_output` defaults to `method="json_schema"`, which sends a
   JSON schema as Ollama's `format` parameter. The cloud endpoint **does not
   enforce it** — asked to return `{"step": "screener"|"risk"|"journal"}` the
   model answered with prose, then with JSON of its own invention. Tool calling
   *is* enforced, so `common.structured()` forces the function-calling path.

3. **Function calling still needs two safety nets.** The model sometimes answers
   the question instead of calling the tool (the stock helper then returns
   `None` with no error), and sometimes calls it with the wrong shape (e.g. a
   list of strings where `list[Section]` was required). `structured()` therefore
   adds a system message that demands the tool call, retries on a missing call,
   and retries with the Pydantic validation error appended when the arguments do
   not match.

4. **Thinking models, lenient critics, and a hard-rule backstop.** `gpt-oss`
   runs a reasoning phase by default; `reasoning=False` removes that
   latency/cost. And an LLM critic alone approves almost anything, which would
   make pattern 06's cycle dead code — so script 06 grades with an explicit
   rubric *and* re-checks the plan in code (stop below entry, R ≥ 2, risk inside
   the 1% budget), and caps the loop at `MAX_ATTEMPTS` before escalating to a
   human instead of blowing the recursion limit. It streams node updates so the
   back-edge is visible
   (`llm_call_generator → llm_call_evaluator → llm_call_generator → …`).

Smaller adjustments: imports come from `langchain_core.messages`
(`langchain.messages` also re-exports them in 1.x); `ToolMessage` content is
stringified in 07; the worker prompt in 05 repeats the watchlist, because a
worker that only sees its section description will otherwise invent tickers and
prices; `recursion_limit` is set on 06/07 so a runaway loop fails loudly
instead of hanging.

## Versions used

```
langgraph 1.2.18   langchain 1.4.7   langchain-core 1.7.0
langchain-ollama 1.1.0   ollama 0.6.3   python 3.14.4
```
