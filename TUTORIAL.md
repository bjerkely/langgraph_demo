# A beginner's tutorial to this repository

This repo is a set of seven tiny pattern programs that each demonstrate one way
to organise calls to a large language model (LLM), plus one longer exercise
(`08_event_log.py`) on a machine event log. They all use the same domain —
planning a $50k swing trade — so you can compare the shapes without learning new
vocabulary every time.

If you have never used an LLM beyond a chat window, start at Part 0. If you have
used ChatGPT but never built anything, start at Part 1. If you only want to run
the code, jump to [Part 7](#part-7--run-it).

```
Part 0  What an LLM actually is
Part 1  The two building blocks: structured output and tools   (01)
Part 2  What a "graph" is, and why anyone would want one
Part 3  The seven patterns, one at a time                      (02-07)
Part 4  Reading LangGraph code: the six calls you need
Part 5  Prompt engineering lessons baked into this repo
Part 6  The gotchas: what actually breaks in production
Part 7  Run it
Part 8  Exercises (includes 08, the event-log exercise)
Part 9  Glossary
```

---

## Part 0 — What an LLM actually is

An LLM is a function:

```
text in  ->  text out
```

That's it. It has no memory between calls. When a chat app feels like it
remembers you, the app is re-sending the whole conversation every time.

Three consequences drive everything in this repo:

1. **The model is stateless.** If step 2 needs to know what step 1 produced, you
   must pass step 1's output into step 2's prompt. That is why every script keeps
   a `State` dictionary.
2. **The model is not a calculator.** It predicts plausible text. Ask it for
   `floor(50000 * 0.01 / 6.5)` and it will often be subtly wrong. Ask it to *call
   a Python function* that computes it, and it will be exactly right. That is
   what "tools" are for.
3. **The model is not deterministic.** `temperature=0` makes it much more
   repeatable, but it is still a probability distribution. Any rule that must
   always hold (risk limits, schema validity, loop termination) belongs in Python,
   not in a prompt.

The model here is `gpt-oss:120b-cloud`, reached over HTTPS through
[Ollama Cloud](https://ollama.com). Nothing is downloaded and nothing runs
locally; `common.get_llm()` is the only place that knows this.

---

## Part 1 — The two building blocks (script 01)

Every workflow and every agent in this repo is made of two tricks. Both appear in
`01_augmentations.py`, before any graph exists.

### 1.1 Structured output: prose in, typed object out

A normal reply is a paragraph. A paragraph is hard to branch on:

```python
msg = llm.invoke("I like NVDA on a pullback to the 20-EMA over two weeks")
# msg.content == "Great choice! NVDA has been..."
```

Instead, declare what you want as a Pydantic model:

```python
class TradeIntent(BaseModel):
    symbol: str
    setup: Literal["breakout", "pullback", "mean_reversion", "trend_continuation"]
    holding_period: str | None = None
    notes: str | None = None

intent = structured(TradeIntent).invoke("I like NVDA on a pullback ...")
# TradeIntent(symbol='NVDA', setup='pullback', holding_period='2 weeks', ...)
```

Now `intent.setup` is a value you can compare, store, and validate — not a
sentence you have to parse.

> Note: `structured()` lives in `common.py`. It is *not* the plain
> `llm.with_structured_output(...)` from the docs. Part 6 explains why.

**The mental model:** structured output is how you get data *out* of an LLM and
into the rest of your program.

> At scale the problem inverts. With a 600-line machine log you cannot ask the
> model to read everything, so you **structure the input first** — parse and
> aggregate in Python — and ask the model only for judgement.
> `08_event_log.py` is that version of this section.

### 1.2 Tool calling: the model asks Python to do the arithmetic

You give the model a normal Python function with a docstring:

```python
@tool
def position_size(account_equity: float, risk_pct: float, entry: float, stop: float) -> str:
    """Size a long swing trade. Always use this tool instead of doing the math."""
    ...
```

Then bind it:

```python
llm_with_tools = llm.bind_tools([position_size])
msg = llm_with_tools.invoke("I have $50,000 and risk 1%. Entry 182.50, stop 176.00.")
```

The model does **not** run the function. It replies with a *request*:

```python
msg.content      # ''  -- it said nothing
msg.tool_calls   # [{'name': 'position_size', 'args': {'account_equity': 50000, ...}}]
```

Your code then runs it:

```python
position_size.invoke(call["args"])
# '76 shares, $6.50 risk per share, $494.00 total risk (13,870.00 notional)'
```

**The mental model:** tool calling is how you get data *into* an LLM's reasoning
without letting it invent numbers. The model chooses *which* function and *what
arguments*; Python does the work.

Who executes the tool? In script 01, you do, in a `for` loop. In script 07, a
graph node does it. That difference is the whole difference between "an LLM call"
and "an agent".

---

## Part 2 — What a "graph" is, and why anyone would want one

Once a task has more than one LLM call, you need to answer: *what runs next?*

You could write:

```python
a = step_a(x)
b = step_b(a)
c = step_c(b)
```

That works until you need a branch ("only if the plan is incomplete"), a loop
("retry until approved"), or fan-out ("run three analysts at once"). Then the
function calls become a tangle of `if`s, and you lose the ability to see the
control flow.

LangGraph makes the control flow **explicit data**:

| Concept | What it is | In this repo |
|---|---|---|
| **State** | A `TypedDict` shared by all nodes | `class State(TypedDict)` |
| **Node** | A Python function `state -> partial update` | `def draft_plan(state): return {"plan": ...}` |
| **Edge** | "After A, always run B" | `add_edge("draft_plan", "add_risk_rules")` |
| **Conditional edge** | A function that returns the next node's name | `add_conditional_edges(..., check_completeness, {...})` |
| **`START` / `END`** | Virtual entry and exit nodes | `add_edge(START, "draft_plan")` |
| **Compile** | Validates and freezes the graph | `graph = builder.compile()` |

A node never mutates the state in place. It returns a **partial dict**, and
LangGraph merges it into the state. That merge is why parallel branches work.

Here is the smallest complete graph in the repo, stripped down:

```python
class State(TypedDict):
    symbol: str
    plan: str

def draft(state: State):
    return {"plan": llm.invoke(f"Write a trade plan for {state['symbol']}").content}

builder = StateGraph(State)
builder.add_node("draft", draft)
builder.add_edge(START, "draft")
builder.add_edge("draft", END)

graph = builder.compile()
graph.invoke({"symbol": "NVDA"})   # -> {'symbol': 'NVDA', 'plan': '...'}
```

That is genuinely all there is. Everything else is variations on *where the
edges point*.

### Fan-out, fan-in, and cycles

```text
Fan-out (03):   START -> A, B, C        add_edge(START, "a") three times
Fan-in  (03):   A, B, C -> joiner       every branch ends at the same node
Cycle   (06):   A -> B -> A             a conditional edge points backwards
Dynamic (05):   Send(...) x N           the number of branches is a runtime value
```

LangGraph runs every node whose incoming edges are satisfied. Parallelism is
therefore *implicit*: you do not write threads. You just draw two edges from
`START` and the framework schedules both.

### Reducers: why two branches can write to one key

If two parallel nodes both return `{"result": ...}`, whose value wins? By
default the last write wins. To collect instead, annotate the key:

```python
completed_sections: Annotated[list, operator.add]
```

`operator.add` means "when a node returns a list for this key, concatenate it."
Script 05 uses this so N workers can each append their section. This is the one
LangGraph concept that surprises people; if you remember one thing from Part 2,
make it this.

---

## Part 3 — The seven patterns

Each script answers a different question. Read them in the order below; it goes
from "one call" to "no fixed path at all".

### 01 — Augmentations: the two primitives
`01_augmentations.py` — no graph yet. Structured output and one tool call.
**Question it answers:** *how do I get typed data out of an LLM, and how do I stop
it doing math?*

### 02 — Prompt chaining: a fixed sequence with a gate
`02_prompt_chaining.py`

```text
START -> draft_plan -> check_completeness --Pass--> END
                            |
                          Fail
                            v
                    add_risk_rules -> add_invalidation -> END
```

Three LLM calls in a row, but the middle one only runs if needed. The decision
is made by **Python**:

```python
def check_completeness(state):
    plan = state["plan"].lower()
    has_levels = all(w in plan for w in ("entry", "stop", "target"))
    return "Pass" if has_levels else "Fail"
```

**Question it answers:** *how do I chain calls without paying for steps I don't
need?*

**Lesson:** the branch condition is deterministic code. An LLM could have judged
completeness, but code is cheaper, faster, and testable. Use the model for
judgement, code for rules.

### 03 — Parallelization: three opinions at once
`03_parallelization.py`

```text
          -> call_llm_1 (technicals) --
START --> call_llm_2 (catalyst)   --> aggregator -> END
          -> call_llm_3 (risk)      --
```

Three independent prompts, no data dependency between them, so they run
concurrently. The aggregator is plain Python string joining — it does not need
an LLM.

**Question it answers:** *how do I cut latency on independent subtasks?*

**Lesson:** if you find yourself writing "first check X, then check Y, then check
Z" where X/Y/Z don't feed each other, fan out.

### 04 — Routing: let the model pick the department
`04_routing.py`

```text
START -> llm_call_router --screener--> screener -> END
                           --risk-----> risk_desk -> END
                           --journal--> journal   -> END
```

The router is a structured-output call over a one-field schema:

```python
class Route(BaseModel):
    step: Literal["screener", "risk", "journal"]
```

`route_decision` maps that string to a node name. The mapping is explicit in the
`add_conditional_edges` dictionary, which doubles as documentation.

**Question it answers:** *how do I handle several kinds of request with one
entry point?*

**Lesson:** classification is the cheapest, most reliable thing an LLM does.
Prefer one small classification call over one giant prompt that tries to do all
three jobs.

### 05 — Orchestrator-worker: N branches, N decided at runtime
`05_orchestrator_worker.py`

```text
START -> orchestrator --Send(section) x N--> llm_call (worker) -> synthesizer -> END
```

This is the first graph whose *width* is unknown when you build it. The
orchestrator asks the model to plan a report as a list of `Section(name,
description)` objects. Then:

```python
def assign_workers(state):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]
```

`Send` launches one worker per section, each with its own small input. Workers
append to the shared list via the `operator.add` reducer, and the synthesizer
joins them.

**Question it answers:** *how do I fan out when the number of subtasks comes from
the model itself?*

**Lessons:**
- `Send` is `map()` for graphs.
- Workers only receive their own `Send` payload, not the full state. That is a
  feature (isolation) and a trap (a worker that can't see the watchlist will
  invent tickers). Script 05 repeats the market data in the worker prompt for
  exactly this reason.

### 06 — Evaluator-optimizer: the first cycle
`06_evaluator_optimizer.py`

```text
START -> llm_call_generator -> llm_call_evaluator --Accepted--> END
             ^                        |
             |                        | Rejected + Feedback
             +------------------------+
                                      | Escalated (after MAX_ATTEMPTS)
                                      v
                                     END
```

A drafter writes a structured `TradePlan`; a risk manager returns
`Review(grade, feedback)`. Rejected plans go back to the drafter **with the
feedback appended to state**, so the next draft is a revision, not a re-roll.

Three details make this pattern safe rather than a toy:

1. **A hard-rule backstop.** Even if the reviewer says "approved", Python
   recomputes and vetoes:

   ```python
   if plan["stop"] >= plan["entry"]:            violations.append(...)
   if r < 2.0:                                  violations.append(...)
   if risk_dollars > ACCOUNT_EQUITY * RISK_PCT * 1.05: violations.append(...)
   ```

2. **A bounded loop.** `MAX_ATTEMPTS = 4`, then the route returns `"Escalated"`
   and the plan goes to a human. `recursion_limit=20` catches anything else.

3. **Visible iteration.** The script streams node updates instead of invoking,
   so you can watch the back-edge fire:

   ```text
   path: llm_call_generator -> llm_call_evaluator -> llm_call_generator -> ... 
   ```

**Question it answers:** *how do I iterate until the output meets a standard?*

**Lesson:** an LLM critic on its own approves almost anything. "LLM judges" only
work when the rubric is explicit and code can verify the hard constraints.

### 07 — Agent: the model owns the control flow
`07_agent.py`

```text
START -> llm_call --has tool_calls--> tool_node
             ^                            |
             +----------------------------+
             |
             +--no tool_calls--> END
```

This graph is only two nodes and one loop, yet it is the most powerful pattern
because the *path* is chosen at runtime:

```text
Human: "I'm long NVDA from 178.40, stop 171.90, target 191.00. Check the quote,
        size the position, and tell me the R-multiple."

AI  -> tool call: quote(NVDA)
Tool -> NVDA: last 178.40, ATR(14) 5.80
AI  -> tool call: position_size(50000, 0.01, 178.4, 171.9)
Tool -> 76 shares, $6.50 risk per share, $494.00 total risk
AI  -> tool call: reward_to_risk(178.4, 171.9, 191.0)
Tool -> R-multiple 1.94
AI  -> final prose answer
```

`should_continue` is the only decision:

```python
def should_continue(state):
    return "tool_node" if state["messages"][-1].tool_calls else END
```

It uses `MessagesState`, LangGraph's built-in state with a message list and the
right reducer already wired up.

**Question it answers:** *what if I can't know the sequence of steps in advance?*

**Lesson:** an agent is not magic. It is a `while` loop around "ask the model; if
it requested tools, run them and ask again." Its quality is bounded by the tools
you give it and the prompt that tells it when to stop.

### Which pattern should I use?

```text
Do I know the sequence of steps in advance?
├── Yes -> Is the output gradeable against a standard?
│          ├── Yes -> 06 evaluator-optimizer
│          └── No  -> Is the number of subtasks fixed?
│                    ├── Yes, independent -> 03 parallelization
│                    ├── Yes, sequential   -> 02 prompt chaining
│                    └── No, comes from the model -> 05 orchestrator-worker
└── No  -> Does the input need classifying first?
           ├── Yes -> 04 routing (a workflow with one model decision)
           └── No  -> 07 agent
```

Rule of thumb from the README: **predefined order -> workflow (02-06); the
model picks the path -> agent (07).** Reach for the simplest shape that works;
agents are the hardest to debug.

---

## Part 4 — Reading LangGraph code: the six calls you need

Every script after 01 has the same skeleton. Learn these six lines and you can
read all of them.

```python
from langgraph.graph import END, START, StateGraph

class State(TypedDict):        # 1. declare the shape of the shared state
    symbol: str
    plan: str

def node(state: State):        # 2. a node: read state, return a partial update
    return {"plan": "..."}

builder = StateGraph(State)    # 3. create a builder bound to that state
builder.add_node("node", node) # 4a. register nodes
builder.add_edge(START, "node")# 4b. wire edges (plain or conditional)
graph = builder.compile()      # 5. validate + freeze -> a Runnable
result = graph.invoke({...})   # 6. run it, get the final state
```

Things worth knowing beyond the skeleton:

| Topic | Detail |
|---|---|
| Input to `invoke` | A dict with the *initial* state. Keys you omit must be optional or absent. |
| Node return | A **partial** dict. Never mutate `state` itself. |
| Streaming | `graph.stream(input, stream_mode="updates")` yields `{node_name: delta}` after each step. Script 06 uses this to show the loop. |
| Recursion limit | Default is 25 super-steps. Cycles need headroom; 06 and 07 set it explicitly so runaway loops *fail loudly* instead of hanging. |
| Diagrams | `common.show_graph(graph, name)` writes Mermaid source to `graphs/<name>.mmd`. |
| Same graph, many inputs | Compilation is one-time; call `.invoke()` repeatedly with different state. Script 02 runs two symbols through one graph. |

---

## Part 5 — Prompt engineering lessons baked into this repo

Read the prompts in the scripts and you'll notice a consistent style. These are
not decoration; each one fixes an observed failure.

1. **Give the model the data, not the instruction to fetch it.** Every prompt
   contains a literal `market_snapshot(...)` table. No retrieval failure, no
   hallucinated price, no rate limit.
2. **Constrain the shape of the answer.** `"In 6 lines: ..."`, `"Use ONLY these
   tickers and prices"`, `"Include no preamble"`. Unconstrained models write
   essays; essays cost tokens and bury the answer.
3. **Assign a role that matches the check.** "Act as a technical analyst" and
   "Act as a risk manager" produce noticeably different emphasis from the same
   model.
4. **Repeat critical context at every hop.** The worker prompt in 05 repeats the
   watchlist because isolation cuts both ways.
5. **Put rules where they can be enforced.** The risk budget is stated in the
   prompt *and* recomputed in Python. Prompt-only rules are suggestions.
6. **Use structured output for anything you will branch on.** Free-text
   classification forces you to parse prose; a `Literal` field does not.
7. **Keep temperature at 0 for this kind of work.** Creativity is not the goal;
   consistency is.

---

## Part 6 — The gotchas: what actually breaks

These are documented in the main README because each was verified against the
live endpoint. They are the most transferable part of the repo.

### 6.1 Endpoint and auth are not the docs' defaults
The LangGraph docs assume a local Ollama at `http://localhost:11434`. This repo
talks to `https://ollama.com`, which needs a bearer token. `ChatOllama`'s
`base_url` does not carry auth, so `common.get_llm()` passes an
`Authorization` header through `client_kwargs`.

### 6.2 `with_structured_output` can silently not be structured
Its default strategy is `method="json_schema"`, which sends a JSON schema as
Ollama's `format` parameter. The cloud endpoint does not enforce it: asked for
`{"step": "screener" | "risk" | "journal"}` the model answered in prose, then
with JSON of its own invention.

**Fix:** force the function-calling path, which *is* enforced, and validate the
arguments yourself. That is what `common.structured(schema)` does.

### 6.3 Models sometimes answer instead of calling the tool
With function calling, a model can decide to just reply, yielding no tool call
and (in the stock API) `None` with no error. It can also call the tool with the
wrong shape, e.g. a list of strings where `list[Section]` was required.

`structured()` therefore:
- prepends a system message demanding the tool call,
- checks that a call actually came back, re-prompting if not,
- catches `ValidationError` and retries with the error text appended.

### 6.4 Thinking models cost latency you may not want
`gpt-oss` runs a reasoning phase by default. For these short nodes that is pure
waiting, so `get_llm()` sets `reasoning=False`. If a task genuinely needs
deliberation, turn it back on and budget for it.

### 6.5 Lenient critics make cycles dead code
Without the hard-rule backstop in 06, the evaluator approves almost everything
and the loop never runs. Always pair an LLM judge with a checkable rubric.

### 6.6 The general lesson
> Use the model for language and judgement. Use code for arithmetic, schemas,
> limits, and exits.

Every workaround above is an instance of that sentence.

---

## Part 7 — Run it

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then put your real OLLAMA_API_KEY in .env
```

`.env` needs:

```ini
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=...          # https://ollama.com/settings/keys
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_TIMEOUT_SECONDS=600
ACCOUNT_EQUITY=50000        # optional
RISK_PCT=0.01               # optional
```

`.env` is gitignored. Never commit it.

### Run one script

```bash
.venv/bin/python 01_augmentations.py
```

Each script prints its output and writes `graphs/<name>.mmd`.

### Run everything

```bash
./run_all.sh                # all seven, one log per script in logs/
```

### See the diagrams

```bash
npx -y @mermaid-js/mermaid-cli -i graphs/07_agent.mmd -o 07_agent.png
```

Or paste the `.mmd` contents into <https://mermaid.live>.

### Suggested reading order

```text
01 -> 02 -> 04 -> 03 -> 05 -> 06 -> 07
```

That order goes: one call, then a chain, then a branch, then a fan-out, then a
dynamic fan-out, then a loop, then full agent autonomy. Each step adds exactly
one new idea.

---

## Part 8 — Exercises

Do these in a scratch copy; they are ordered by difficulty.

1. **Change the domain.** Rewrite `02_prompt_chaining.py` to draft a study plan
   instead of a trade plan. The graph should not change at all — proof that the
   pattern is independent of the domain.
2. **Add a gate.** In `04_routing.py`, add a fourth route for "general
   questions". Notice that adding a route is three edits: schema literal,
   routing function, edge mapping.
3. **Break the reducer.** In `05_orchestrator_worker.py`, remove
   `Annotated[list, operator.add]` from `completed_sections` and run it. Observe
   that only one worker's section survives. Put it back.
4. **Make the loop fail.** In `06_evaluator_optimizer.py`, set
   `MAX_ATTEMPTS = 1` and tighten the hard rules (e.g. `r < 5.0`). Watch the
   escalation path and the streamed path.
5. **Give the agent a bad tool.** In `07_agent.py`, add a `quote` variant that
   returns a string for an unknown ticker, and ask about a ticker that isn't in
   `SAMPLE_MARKET`. Does the model recover, invent, or stop?
6. **Remove the safety net.** Point script 04 at
   `llm.with_structured_output(Route)` instead of `structured(Route)` and run it
   a few times. This is the fastest way to feel why Part 6.2 exists.
7. **Make the event log yours** (`08_event_log.py`). The script turns a
   lithography tool's event log into a verified incident report. It runs on the
   synthetic fixture; the exercise is to make it work on a log it was not written
   for. Five tiers, in order:
   - **Format.** Write a `FormatSpec` for your log — copy
     `formats/example_custom.json`: a regex with named groups `ts`, `subsystem`,
     `severity`, `event_id`, plus a payload style. Run `--dry-run` until the skip
     count is small and the sampled lines look right.
   - **Scale.** Change `--group-window` and watch the group count. Too fine and
     the prompt gets expensive; too coarse and two incidents merge. The
     aggregation key is the design decision, not the prompt.
   - **Safety net.** Delete the `ids exist in log` check in `verify()` and run a
     few times. A model that never saw a line will still cite a plausible event
     id. Watch it pass, then restore the check.
   - **Graph.** The pipeline is `aggregate → label → correlate → verify`. Wrap it
     in a `StateGraph` with a conditional edge back to `correlate` when a check
     fails, reusing 06's cycle and `MAX_ATTEMPTS`. The loop is then justified by a
     checkable rubric instead of an LLM's opinion.
   - **Stretch.** Use structured output to induce the `FormatSpec` itself from 20
     sample lines, cache the JSON, and parse deterministically afterwards.

   > At scale, structure the input before you structure the output: parse and
   > count in Python, ask the model only for meaning.

---

## Part 9 — Glossary

| Term | Meaning in this repo |
|---|---|
| **LLM** | The text-in/text-out model (`gpt-oss:120b-cloud`). |
| **Prompt** | The text you send. A list of `SystemMessage` + `HumanMessage` is a prompt with roles. |
| **System message** | Standing instructions ("you are a risk manager"). |
| **Structured output** | Forcing the reply into a Pydantic schema instead of prose. |
| **Event log** | Timestamped records emitted by a machine; many lines, far fewer distinct events. |
| **Aggregation** | Collapsing repeated events into one row (count, first/last, min/max) before prompting. |
| **Tool / function calling** | The model requests a Python function with arguments; your code runs it. |
| **Agent** | A loop where the *model* decides which tool runs next. |
| **Workflow** | A graph whose path is fixed in code, with model calls inside the nodes. |
| **State** | The dict threaded through every node. |
| **Node** | A function that reads state and returns a partial update. |
| **Edge** | A transition. Conditional edges choose based on state. |
| **Reducer** | How parallel writes to one state key are merged (`operator.add` = concatenate). |
| **Super-step** | One round of node execution; `recursion_limit` counts these. |
| **Setup** | `breakout`, `pullback`, `mean_reversion`, `trend_continuation`. |
| **Entry / stop / target** | The three prices every usable trade plan states. |
| **R-multiple** | `(target - entry) / (entry - stop)`. The desk requires >= 2.0. |
| **Position size** | `floor(equity * risk_pct / (entry - stop))` shares. |
| **ATR(14)** | Average true range; stops sit ~1.5x ATR away so noise doesn't hit them. |
| **Invalidation** | Evidence that the thesis is wrong *before* the stop is reached. |

---

## Where to go next

- The upstream guide: <https://docs.langchain.com/oss/python/langgraph/workflows-agents>
- Add persistence (`checkpointer`) so a graph can resume after a crash.
- Add `interrupt()` for human-in-the-loop approval on pattern 06's escalation.
- Run `08_event_log.py` against a real machine log and write a `FormatSpec` for it.
- Add a real data tool to pattern 07 and watch how the agent's behaviour changes
  when a tool can fail.
